"""Regression tests for the IAFv3-to-GQR refactor.

The full repository imports several optional training backbones at package
import time.  This test loads ``mmvg_fusion.py`` with only the unused backbone
imports stubbed, so it validates the real fusion methods without downloading a
foundation checkpoint.
"""

import importlib.util
import sys
import types
from pathlib import Path

import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_fusion_module():
    package_name = "_tsar_test_models"
    package = types.ModuleType(package_name)
    package.__path__ = [str(REPO_ROOT / "models")]
    sys.modules[package_name] = package

    clip_stub = types.ModuleType(f"{package_name}.clip")
    sys.modules[f"{package_name}.clip"] = clip_stub
    vl_stub = types.ModuleType(f"{package_name}.vl_transformer")
    vl_stub.build_vl_transformer = lambda args: None
    sys.modules[f"{package_name}.vl_transformer"] = vl_stub

    tsar_spec = importlib.util.spec_from_file_location(
        f"{package_name}.tsar_modules",
        REPO_ROOT / "models" / "tsar_modules.py",
    )
    tsar_module = importlib.util.module_from_spec(tsar_spec)
    sys.modules[tsar_spec.name] = tsar_module
    tsar_spec.loader.exec_module(tsar_module)

    fusion_spec = importlib.util.spec_from_file_location(
        f"{package_name}.mmvg_fusion",
        REPO_ROOT / "models" / "mmvg_fusion.py",
    )
    fusion_module = importlib.util.module_from_spec(fusion_spec)
    sys.modules[fusion_spec.name] = fusion_module
    fusion_spec.loader.exec_module(fusion_module)
    return fusion_module, tsar_module


def _legacy_iafv3(model, rgb_tokens, ir_tokens, rgb_image):
    """The pre-refactor IAFv3 math, retained as a test-only golden path."""
    cls_rgb, map_rgb = model._tokens_to_map(rgb_tokens)
    cls_ir, map_ir = model._tokens_to_map(ir_tokens)
    illum = model._illumination_prior(rgb_image, map_rgb.shape[-2:])
    illum_token = illum.flatten(2).permute(0, 2, 1)
    illum_conf = 1.0 - torch.clamp(4.0 * (illum_token - 0.5) ** 2, min=0.0, max=1.0)

    rgb_patch = map_rgb.flatten(2).permute(0, 2, 1)
    ir_patch = map_ir.flatten(2).permute(0, 2, 1)
    rgb_norm = torch.nn.functional.layer_norm(rgb_patch, (rgb_patch.shape[-1],))
    ir_norm = torch.nn.functional.layer_norm(ir_patch, (ir_patch.shape[-1],))
    diff_patch = torch.abs(rgb_norm - ir_norm)
    token_in = torch.cat([rgb_norm, ir_norm, diff_patch], dim=-1)
    temperature = torch.clamp(model.iafv3_temp, min=0.5, max=2.0)
    token_gate = torch.sigmoid(model.iafv3_token_gate(token_in) / temperature)

    rgb_global = rgb_norm.mean(dim=1)
    ir_global = ir_norm.mean(dim=1)
    channel_input = torch.cat([rgb_global, ir_global, torch.abs(rgb_global - ir_global)], dim=-1)
    channel_gate = torch.sigmoid(model.iafv3_channel_gate(channel_input)).unsqueeze(1)
    channel_bias = channel_gate.mean(dim=-1, keepdim=True)
    w_rgb = torch.sigmoid(
        model.iafv3_alpha * (illum_token - 0.5) * illum_conf
        + model.iafv3_beta * (token_gate - 0.5)
        + model.iafv3_gamma * (channel_bias - 0.5)
    )

    fused_patch = w_rgb * rgb_patch + (1.0 - w_rgb) * ir_patch
    fused_patch = fused_patch + torch.tanh(model.iafv3_delta) * channel_gate * (rgb_patch - ir_patch)
    fused_map = fused_patch.permute(0, 2, 1).reshape_as(map_rgb)
    cls_weight = w_rgb.mean(dim=1, keepdim=True)
    fused_cls = cls_weight * cls_rgb + (1.0 - cls_weight) * cls_ir
    return model._map_to_tokens(fused_cls, fused_map)


def _fusion_harness(fusion_module, tsar_module, dim):
    model = object.__new__(fusion_module.MMVGFusion)
    nn.Module.__init__(model)
    model.hidden_dim = dim
    model.iafv3_token_gate = nn.Sequential(nn.Linear(dim * 3, dim), nn.ReLU(), nn.Linear(dim, 1))
    model.iafv3_channel_gate = nn.Sequential(
        nn.Linear(dim * 3, dim), nn.ReLU(), nn.Linear(dim, dim)
    )
    model.iafv3_alpha = nn.Parameter(torch.tensor(1.0))
    model.iafv3_beta = nn.Parameter(torch.tensor(0.9))
    model.iafv3_gamma = nn.Parameter(torch.tensor(0.6))
    model.iafv3_delta = nn.Parameter(torch.tensor(0.2))
    model.iafv3_temp = nn.Parameter(torch.tensor(1.0))
    model.rgb_aux_head = tsar_module.TextConditionedModalityHead(dim)
    model.tir_aux_head = tsar_module.TextConditionedModalityHead(dim)
    model.gqr = tsar_module.GroundingQualityRouter(dim)
    model.gqr_eta_raw = nn.Parameter(torch.tensor(0.0))
    model.gqr_eta_max = 2.0
    model._gqr_correction_scale = 1.0
    model._tsar_aux = {}
    return model


def test_iafv3_refactor_and_zero_init_gqr_are_equivalent():
    fusion_module, tsar_module = _load_fusion_module()
    torch.manual_seed(17)
    batch_size, patch_side, dim = 2, 4, 16
    token_count = patch_side * patch_side + 1
    model = _fusion_harness(fusion_module, tsar_module, dim)
    rgb_tokens = torch.randn(token_count, batch_size, dim)
    ir_tokens = torch.randn(token_count, batch_size, dim)
    rgb_image = torch.randn(batch_size, 3, 32, 32)
    text_embed = torch.randn(batch_size, dim)

    legacy = _legacy_iafv3(model, rgb_tokens, ir_tokens, rgb_image)
    refactored = model._fusion_iafv3(rgb_tokens, ir_tokens, rgb_image)
    model._tsar_aux = {"thermal_rho": torch.tensor(0.5)}
    gqr = model._fusion_gqrv1(rgb_tokens, ir_tokens, rgb_image, text_embed)

    assert torch.allclose(refactored, legacy, atol=1e-6, rtol=1e-5)
    assert torch.allclose(gqr, refactored, atol=1e-6, rtol=1e-5)
    assert model._tsar_aux["rgb_aux_box"].shape == (batch_size, 4)
    assert model._tsar_aux["tir_aux_box"].shape == (batch_size, 4)
    assert model._tsar_aux["router_logits"].shape == (batch_size, 2)
    assert model._tsar_aux["thermal_rho"].item() == 0.5
    assert torch.allclose(
        model._tsar_aux["router_prob"].sum(dim=-1),
        torch.ones(batch_size),
        atol=1e-6,
    )

    # The InfMAE variant deliberately retains the proven GQR fusion path; it
    # only changes the preceding TIR-token enhancement route.
    model.fusion_method = "GQRv1InfMAE"
    infmae_dispatch = model._fuse_visual_tokens(
        rgb_tokens,
        ir_tokens,
        rgb_image,
        text_embed,
    )
    assert torch.allclose(infmae_dispatch, gqr, atol=1e-6, rtol=1e-5)

    # The direct InfMAE frontend replaces only the TIR token producer.  Once
    # its tokens reach the original fusion boundary, it must use exact IAFv3
    # math rather than implicitly enabling any GQR route.
    model.fusion_method = "InfMAEDirectV2"
    direct_dispatch = model._fuse_visual_tokens(
        rgb_tokens,
        ir_tokens,
        rgb_image,
    )
    assert torch.allclose(direct_dispatch, refactored, atol=1e-6, rtol=1e-5)

    # Warmup must preserve IAFv3 even after eta itself has changed.
    with torch.no_grad():
        model.gqr_eta_raw.fill_(0.7)
    model.set_gqr_correction_scale(0.0)
    warmup_gqr = model._fusion_gqrv1(rgb_tokens, ir_tokens, rgb_image, text_embed)
    assert torch.allclose(warmup_gqr, refactored, atol=1e-6, rtol=1e-5)
    assert model._tsar_aux["gqr_eta"].item() == 0.0
    assert model._tsar_aux["gqr_correction_scale"].item() == 0.0

    main_out = tuple(torch.empty(0) for _ in range(5))
    model.enable_gqr = True
    model.train()
    assert len(model._format_model_output(main_out)) == 6
    model.eval()
    assert len(model._format_model_output(main_out)) == 5
    model.enable_gqr = False
    model.train()
    assert len(model._format_model_output(main_out)) == 5


def test_direct_infmae_keeps_rgb_lora_trainable_from_stage_zero():
    fusion_module, _ = _load_fusion_module()
    model = object.__new__(fusion_module.MMVGFusion)
    nn.Module.__init__(model)
    model.open_lora = True
    model.enable_infmae_direct = True
    model.clip = nn.Module()
    model.clip.encoder = nn.Module()
    model.clip.encoder.layers = nn.ModuleList([nn.Module()])
    model.clip.encoder.layers[0].lora_A = nn.Linear(2, 2, bias=False)
    model.clip.encoder.layers[0].lora_B = nn.Linear(2, 2, bias=False)

    model._set_lora_stage_trainable(0)
    assert all(parameter.requires_grad for parameter in model.clip.parameters())

    # Existing modes retain the original staged HiLoRA behavior.
    model.enable_infmae_direct = False
    model._set_lora_stage_trainable(0)
    assert all(not parameter.requires_grad for parameter in model.clip.parameters())
