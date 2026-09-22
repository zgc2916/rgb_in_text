"""Regression tests for the frozen InfMAE thermal-expert path.

These tests deliberately target the model-only integration contract: the
InfMAE checkpoint must load exactly, its input statistics must be reconstructed
from the existing RGBT tensor, and a fresh adapter must be an exact identity
on the established CLIP TIR stream.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_tsar_modules():
    spec = importlib.util.spec_from_file_location(
        "tsar_modules_infmae_test",
        REPO_ROOT / "models" / "tsar_modules.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_fusion_module():
    """Load MMVGFusion without constructing optional foundation backbones."""

    package_name = "_infmae_fusion_test_models"
    package = types.ModuleType(package_name)
    package.__path__ = [str(REPO_ROOT / "models")]
    sys.modules[package_name] = package

    sys.modules[f"{package_name}.clip"] = types.ModuleType(f"{package_name}.clip")
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
    return fusion_module


def test_infmae_input_normalizer_reconstructs_dataset_thermal_values():
    modules = _load_tsar_modules()
    normalizer = modules.InfMAEThermalInputNormalizer("rgbtvg_flir")

    # A zero-valued standardized FLIR TIR channel denotes the dataset mean.
    standardized_tir = torch.zeros(2, 3, 4, 4)
    normalized = normalizer(standardized_tir)
    expected = torch.full_like(normalized, (0.5337 - 0.425) / 0.200)

    assert normalized.shape == (2, 3, 4, 4)
    assert torch.allclose(normalized, expected, atol=1e-6)
    assert torch.equal(normalized[:, 0], normalized[:, 1])
    assert torch.equal(normalized[:, 1], normalized[:, 2])


def test_infmae_adapter_is_exact_identity_at_zero_init_and_then_learns():
    modules = _load_tsar_modules()
    torch.manual_seed(23)
    adapter = modules.InfMAEMultiScaleThermalAdapter(
        f2_dim=4,
        f3_dim=8,
        out_dim=16,
        bottleneck_dim=4,
    )
    tir_tokens = torch.randn(2, 17, 16, requires_grad=True)
    f2 = torch.randn(2, 4, 8, 8)
    f3 = torch.randn(2, 8, 4, 4)

    enhanced, aux = adapter(tir_tokens, f2, f3)
    assert torch.equal(enhanced, tir_tokens)
    assert aux["thermal_rho"].item() == 0.5
    assert aux["thermal_residual_ratio"].item() == 0.0

    # Even at the exact-identity initialization, the zero output projection
    # gets a first-step gradient.  The original CLS token remains untouched.
    enhanced.square().mean().backward()
    assert adapter.adapter_up.weight.grad is not None
    assert torch.count_nonzero(adapter.adapter_up.weight.grad).item() > 0

    with torch.no_grad():
        adapter.adapter_up.weight.normal_(mean=0.0, std=0.02)
    enhanced, _ = adapter(tir_tokens, f2, f3)
    assert torch.equal(enhanced[:, :1], tir_tokens[:, :1])
    enhanced.square().mean().backward()
    assert adapter.f2_proj.weight.grad is not None
    assert adapter.f3_proj.weight.grad is not None
    assert adapter.adapter_up.weight.grad is not None


def test_local_rgb_tir_evidence_adapter_is_identity_then_opens_with_gradient():
    modules = _load_tsar_modules()
    torch.manual_seed(47)
    adapter = modules.LocalCrossModalThermalEvidenceAdapter(
        visual_dim=8,
        num_feature_levels=2,
        bottleneck_dim=4,
        neighborhood_size=3,
    )
    rgb_features = [torch.randn(2, 17, 8, requires_grad=True) for _ in range(2)]
    tir_features = [torch.randn(2, 17, 8, requires_grad=True) for _ in range(2)]

    enhanced, aux = adapter(rgb_features, tir_features)
    assert all(torch.equal(output, source) for output, source in zip(enhanced, tir_features))
    assert aux['local_evidence_residual_ratio'].item() == 0.0
    assert torch.isfinite(aux['local_evidence_attention_entropy'])
    assert torch.isfinite(aux['local_evidence_attention_peak'])

    sum(output.square().mean() for output in enhanced).backward()
    assert all(projection.weight.grad is not None for projection in adapter.residual_ups)
    assert all(
        torch.count_nonzero(projection.weight.grad).item() > 0
        for projection in adapter.residual_ups
    )

    with torch.no_grad():
        adapter.residual_ups[0].weight.normal_(mean=0.0, std=0.02)
    enhanced, _ = adapter(rgb_features, tir_features)
    assert torch.equal(enhanced[0][:, :1], tir_features[0][:, :1])
    assert not torch.equal(enhanced[0][:, 1:], tir_features[0][:, 1:])
    sum(output.square().mean() for output in enhanced).backward()
    assert adapter.query_projection.weight.grad is not None


def test_infmae_lora_linear_is_identity_at_init_and_trains_only_adapter_weights():
    modules = _load_tsar_modules()
    torch.manual_seed(27)
    base = nn.Linear(8, 12)
    wrapper = modules.InfMAELoRALinear(base, rank=2, alpha=4.0)
    inputs = torch.randn(3, 5, 8, requires_grad=True)

    reference = base(inputs).detach()
    output = wrapper(inputs)
    assert torch.equal(output, reference)

    output.square().mean().backward()
    assert wrapper.lora_B.weight.grad is not None
    assert torch.count_nonzero(wrapper.lora_B.weight.grad).item() > 0
    assert wrapper.lora_A.weight.grad is not None
    assert torch.count_nonzero(wrapper.lora_A.weight.grad).item() == 0
    assert wrapper.base.weight.grad is None

    wrapper.zero_grad(set_to_none=True)
    with torch.no_grad():
        wrapper.lora_B.weight.normal_(mean=0.0, std=0.02)
    wrapper(inputs).square().mean().backward()
    assert wrapper.lora_A.weight.grad is not None
    assert torch.count_nonzero(wrapper.lora_A.weight.grad).item() > 0


def test_infmae_lora_conv_is_identity_at_init_and_trains_only_adapter_weights():
    modules = _load_tsar_modules()
    torch.manual_seed(28)
    base = nn.Conv2d(8, 12, kernel_size=1, bias=True)
    wrapper = modules.InfMAELoRAConv2d(base, rank=2, alpha=4.0)
    inputs = torch.randn(3, 8, 5, 5, requires_grad=True)

    reference = base(inputs).detach()
    output = wrapper(inputs)
    assert torch.equal(output, reference)

    output.square().mean().backward()
    assert wrapper.lora_B.weight.grad is not None
    assert torch.count_nonzero(wrapper.lora_B.weight.grad).item() > 0
    assert wrapper.lora_A.weight.grad is not None
    assert torch.count_nonzero(wrapper.lora_A.weight.grad).item() == 0
    assert wrapper.base.weight.grad is None

    wrapper.zero_grad(set_to_none=True)
    with torch.no_grad():
        wrapper.lora_B.weight.normal_(mean=0.0, std=0.02)
    wrapper(inputs).square().mean().backward()
    assert wrapper.lora_A.weight.grad is not None
    assert torch.count_nonzero(wrapper.lora_A.weight.grad).item() > 0


def test_direct_infmae_adapter_emits_the_original_lavs_token_interface():
    modules = _load_tsar_modules()
    torch.manual_seed(29)
    adapter = modules.InfMAEDirectThermalAdapter(
        f2_dim=4,
        f3_dim=8,
        token_dim=16,
        num_feature_levels=4,
        bottleneck_dim=4,
        f2_scale_init=0.10,
    )
    f2 = torch.randn(2, 4, 8, 8, requires_grad=True)
    f3 = torch.randn(2, 8, 4, 4, requires_grad=True)

    stages, pooled = adapter(f2, f3)

    assert isinstance(stages, list)
    assert len(stages) == 4
    assert all(stage.shape == (2, 17, 16) for stage in stages)
    assert pooled.shape == (2, 16)
    assert torch.allclose(torch.sigmoid(adapter.f2_scale_raw), torch.tensor(0.10))
    # The stage adapters are intentionally distinct trainable heads rather
    # than four aliases of a single InfMAE map.
    assert adapter.stage_adapters[0] is not adapter.stage_adapters[1]

    sum(stage.square().mean() for stage in stages).backward()
    assert adapter.f2_proj.weight.grad is not None
    assert adapter.f3_proj.weight.grad is not None
    assert adapter.stage_adapters[0].up.weight.grad is not None


def test_hierarchical_infmae_adapter_preserves_lavs_interface_and_cross_scale_gradients():
    modules = _load_tsar_modules()
    torch.manual_seed(31)
    adapter = modules.InfMAEHierarchicalThermalAdapter(
        f2_dim=4,
        f3_dim=8,
        token_dim=16,
        num_feature_levels=4,
        bottleneck_dim=4,
        f2_scale_init=0.10,
        transition_mix_init=0.35,
    )
    f2 = torch.randn(2, 4, 8, 8, requires_grad=True)
    f3 = torch.randn(2, 8, 4, 4, requires_grad=True)

    stages, pooled = adapter(f2, f3)

    assert isinstance(stages, list)
    assert len(stages) == 4
    assert all(stage.shape == (2, 17, 16) for stage in stages)
    assert pooled.shape == (2, 16)
    assert torch.allclose(
        torch.sigmoid(adapter.context_gate[-1].bias),
        torch.full((4,), 0.10),
        atol=1e-6,
    )
    # A cascade must produce distinct semantic levels while preserving the
    # original LAVS token shape at every level.
    assert not torch.allclose(stages[0], stages[1])

    sum(stage.square().mean() for stage in stages).backward()
    assert adapter.f2_local_downsample.weight.grad is not None
    assert torch.count_nonzero(adapter.f2_local_downsample.weight.grad).item() > 0
    assert adapter.f2_proj.weight.grad is not None
    assert adapter.f3_proj.weight.grad is not None
    assert adapter.context_gate[-1].bias.grad is not None
    assert adapter.transition_mix_raw.grad is not None
    assert adapter.stage_adapters[0].up.weight.grad is not None


def test_target_aware_token_pool_and_projection_are_differentiable():
    modules = _load_tsar_modules()
    torch.manual_seed(37)
    # A 4x4 token grid mirrors the 14x14 direct-InfMAE token interface at a
    # small test size.  The two boxes include a tiny boundary target to cover
    # the pooler's degenerate-box safeguard.
    tokens = torch.randn(2, 16, 8, requires_grad=True)
    boxes = torch.tensor(
        [[0.50, 0.50, 0.50, 0.50], [0.99, 0.02, 0.001, 0.001]],
        dtype=torch.float32,
    )
    pool = modules.TargetAwareTokenPool(pool_size=3)
    projector = modules.TargetFeatureProjector(8, 5)

    pooled = pool(tokens, boxes)
    projected = projector(pooled)
    assert pooled.shape == (2, 8)
    assert projected.shape == (2, 5)
    assert torch.isfinite(projected).all()

    projected.square().mean().backward()
    assert tokens.grad is not None
    assert torch.count_nonzero(tokens.grad).item() > 0
    assert projector.projection.weight.grad is not None


def test_text_guided_semantic_bridge_is_identity_at_zero_and_trainable_after_warmup():
    modules = _load_tsar_modules()
    torch.manual_seed(41)
    bridge = modules.TextGuidedThermalSemanticBridge(
        visual_dim=8,
        semantic_dim=5,
        temperature=0.2,
    )
    tokens = torch.randn(2, 17, 8, requires_grad=True)
    semantic_tokens = torch.randn(2, 16, 5, requires_grad=True)
    text = torch.randn(2, 5, requires_grad=True)

    # Zero initialization and a zero warmup scale both preserve the exact
    # original thermal token stream, including its CLS token.
    output, stats = bridge(tokens, semantic_tokens, text, residual_scale=1.0)
    assert torch.equal(output, tokens)
    assert stats['semantic_bridge_gain'].item() == 0.0
    blocked, blocked_stats = bridge(tokens, semantic_tokens, text, residual_scale=0.0)
    assert torch.equal(blocked, tokens)
    assert blocked_stats['semantic_bridge_gain'].item() == 0.0

    # The gate itself can open from a grounding loss, while the text teacher
    # remains detached exactly like the A3 target loss.
    weighting = torch.randn_like(output)
    (output * weighting).sum().backward()
    assert bridge.residual_gain.grad is not None
    assert bridge.residual_gain.grad.abs().item() > 0
    assert text.grad is None

    bridge.zero_grad(set_to_none=True)
    tokens_2 = torch.randn(2, 17, 8, requires_grad=True)
    semantic_tokens_2 = torch.randn(2, 16, 5, requires_grad=True)
    text_2 = torch.randn(2, 5, requires_grad=True)
    with torch.no_grad():
        bridge.residual_gain.fill_(0.2)
    opened, opened_stats = bridge(
        tokens_2,
        semantic_tokens_2,
        text_2,
        residual_scale=1.0,
    )
    assert torch.equal(opened[:, :1], tokens_2[:, :1])
    assert not torch.allclose(opened[:, 1:], tokens_2[:, 1:])
    assert opened_stats['semantic_bridge_gain'].abs().item() > 0
    opened.square().mean().backward()
    assert bridge.text_to_visual.weight.grad is not None
    assert semantic_tokens_2.grad is not None
    assert torch.count_nonzero(semantic_tokens_2.grad).item() > 0
    assert text_2.grad is None


def test_target_distribution_and_spatial_bridge_preserve_zero_init_contract():
    modules = _load_tsar_modules()
    torch.manual_seed(47)
    boxes = torch.tensor(
        [[0.50, 0.50, 0.30, 0.25], [0.99, 0.02, 0.001, 0.001]],
        dtype=torch.float32,
    )
    target_distribution = modules.TargetAwareTokenPool.box_to_patch_distribution(
        boxes,
        patch_count=16,
    )
    assert target_distribution.shape == (2, 16)
    assert torch.all(target_distribution >= 0)
    assert torch.allclose(
        target_distribution.sum(dim=-1),
        torch.ones(2),
        atol=1e-6,
    )

    bridge = modules.TextGuidedThermalSpatialBridge(
        visual_dim=8,
        semantic_dim=5,
        temperature=0.2,
    )
    tokens = torch.randn(2, 17, 8, requires_grad=True)
    semantic_tokens = torch.randn(2, 16, 5, requires_grad=True)
    text = torch.randn(2, 5, requires_grad=True)
    output, stats = bridge(tokens, semantic_tokens, text, residual_scale=1.0)
    assert torch.equal(output, tokens)
    attention = stats['spatial_attention']
    assert attention.shape == (2, 16)
    assert torch.allclose(attention.sum(dim=-1), torch.ones(2), atol=1e-6)

    # The box-supervision path must train semantic patch tokens even before
    # the zero-init residual opens, while the CLIP text teacher stays frozen.
    spatial_loss = -attention[:, 5].clamp_min(torch.finfo(attention.dtype).eps).log().mean()
    (output.square().mean() + spatial_loss).backward()
    assert bridge.residual_gain.grad is not None
    assert bridge.residual_gain.grad.abs().item() > 0
    assert semantic_tokens.grad is not None
    assert torch.count_nonzero(semantic_tokens.grad).item() > 0
    assert text.grad is None

    bridge.zero_grad(set_to_none=True)
    tokens_2 = torch.randn(2, 17, 8, requires_grad=True)
    semantic_tokens_2 = torch.randn(2, 16, 5, requires_grad=True)
    text_2 = torch.randn(2, 5, requires_grad=True)
    with torch.no_grad():
        bridge.residual_gain.fill_(0.2)
    opened, _ = bridge(tokens_2, semantic_tokens_2, text_2, residual_scale=1.0)
    assert torch.equal(opened[:, :1], tokens_2[:, :1])
    assert not torch.allclose(opened[:, 1:], tokens_2[:, 1:])
    opened.square().mean().backward()
    assert bridge.text_to_visual.weight.grad is not None
    assert bridge.semantic_to_visual.weight.grad is not None
    assert text_2.grad is None


def test_alignment_gradient_gate_warms_projector_without_changing_tir_features():
    fusion_module = _load_fusion_module()
    thermal_feature = torch.randn(2, 8, requires_grad=True)
    projector = nn.Linear(8, 4)

    gated = fusion_module.MMVGFusion._scale_infmae_alignment_gradient(
        thermal_feature,
        0.0,
    )
    assert torch.equal(gated, thermal_feature.detach())
    projector(gated).square().mean().backward()
    assert thermal_feature.grad is not None
    assert torch.count_nonzero(thermal_feature.grad).item() == 0
    assert projector.weight.grad is not None


def test_grounding_gradient_gate_can_isolate_adapter_from_main_loss():
    fusion_module = _load_fusion_module()
    thermal_feature = torch.randn(2, 8, requires_grad=True)

    # This is the same forward-preserving gate used between the direct
    # InfMAE adapter and LAVS/grounding path.  A zero setting must stop the
    # main grounding loss from changing the adapter, without changing the
    # actual tensor seen downstream.
    gated = fusion_module.MMVGFusion._scale_infmae_alignment_gradient(
        thermal_feature,
        0.0,
    )
    assert torch.equal(gated, thermal_feature.detach())
    gated.square().mean().backward()
    assert thermal_feature.grad is not None
    assert torch.count_nonzero(thermal_feature.grad).item() == 0


@pytest.mark.skipif(
    not (REPO_ROOT / "InfMAE" / "InfMAE.pth").is_file(),
    reason="local InfMAE checkpoint is not available",
)
def test_local_infmae_checkpoint_strictly_exposes_dense_f2_f3_features():
    modules = _load_tsar_modules()
    encoder = modules.FrozenInfMAEEncoder(REPO_ROOT / "InfMAE" / "InfMAE.pth")

    assert encoder.training is False
    assert all(not parameter.requires_grad for parameter in encoder.parameters())
    encoder.train()
    assert encoder.training is False

    with torch.no_grad():
        f2, f3 = encoder(torch.zeros(1, 3, 224, 224))
    assert f2.shape == (1, 384, 28, 28)
    assert f3.shape == (1, 768, 14, 14)
    assert f2.requires_grad is False
    assert f3.requires_grad is False


@pytest.mark.skipif(
    not (REPO_ROOT / "InfMAE" / "InfMAE.pth").is_file(),
    reason="local InfMAE checkpoint is not available",
)
def test_local_infmae_late_lora_preserves_step_zero_features_and_receives_gradients():
    modules = _load_tsar_modules()
    torch.manual_seed(53)
    encoder = modules.FrozenInfMAEEncoder(REPO_ROOT / "InfMAE" / "InfMAE.pth")
    thermal = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        f2_reference, f3_reference = encoder(thermal)

    encoder.enable_late_lora(num_blocks=3, rank=2, alpha=4.0)
    encoder.train()
    f2, f3 = encoder(thermal)
    assert encoder.training is False
    assert encoder.blocks3[7].training is False
    assert encoder.blocks3[8].training is True
    torch.testing.assert_close(f2, f2_reference, atol=0.0, rtol=0.0)
    torch.testing.assert_close(f3, f3_reference, atol=1e-6, rtol=1e-6)
    assert f3.requires_grad is True

    f3.square().mean().backward()
    late_qkv = encoder.blocks3[8].attn.qkv
    assert late_qkv.lora_B.weight.grad is not None
    assert torch.count_nonzero(late_qkv.lora_B.weight.grad).item() > 0
    assert late_qkv.base.weight.grad is None
    assert all(
        parameter.grad is None
        for parameter in encoder.blocks3[7].parameters()
    )


def test_infmae_checkpoint_contract_omits_frozen_expert_but_keeps_adapter():
    fusion_module = _load_fusion_module()
    model = object.__new__(fusion_module.MMVGFusion)
    nn.Module.__init__(model)
    model.enable_infmae = True
    model.infmae_encoder = nn.Linear(3, 3)
    model.infmae_input_normalizer = nn.BatchNorm1d(3)
    model.infmae_adapter = nn.Linear(3, 3)

    checkpoint_state = model.state_dict()
    assert any(key.startswith("infmae_adapter.") for key in checkpoint_state)
    assert not any(key.startswith("infmae_encoder.") for key in checkpoint_state)
    assert not any(key.startswith("infmae_input_normalizer.") for key in checkpoint_state)

    # An original GQR checkpoint lacks all three InfMAE prefixes.  The new
    # branch must still accept it strictly because the local expert checkpoint
    # is loaded independently and the adapter is intentionally fresh.
    incompatible = model.load_state_dict({}, strict=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []


def test_direct_infmae_checkpoint_contract_omits_frozen_encoder_and_legacy_tir_lora():
    fusion_module = _load_fusion_module()
    model = object.__new__(fusion_module.MMVGFusion)
    nn.Module.__init__(model)
    model.enable_infmae = False
    model.enable_infmae_direct = True
    model.infmae_direct_encoder = nn.Linear(3, 3)
    model.infmae_direct_input_normalizer = nn.BatchNorm1d(3)
    model.infmae_direct_adapter = nn.Linear(3, 3)
    model.infmae_direct_cls_projection = nn.Linear(3, 3)

    checkpoint_state = model.state_dict()
    assert any(key.startswith("infmae_direct_adapter.") for key in checkpoint_state)
    assert any(key.startswith("infmae_direct_cls_projection.") for key in checkpoint_state)
    assert not any(key.startswith("infmae_direct_encoder.") for key in checkpoint_state)
    assert not any(key.startswith("infmae_direct_input_normalizer.") for key in checkpoint_state)

    # The official MMVG initializer carries an obsolete TIR LoRA adapter.
    # Direct InfMAE has no such branch, so strict compatibility accepts it.
    incompatible = model.load_state_dict(
        {"clip.base_model.model.fake.lora_ir.weight": torch.ones(1)},
        strict=True,
    )
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []


def test_direct_infmae_checkpoint_keeps_late_lora_but_not_frozen_encoder_weights():
    fusion_module = _load_fusion_module()
    model = object.__new__(fusion_module.MMVGFusion)
    nn.Module.__init__(model)
    model.enable_infmae = False
    model.enable_infmae_direct = True
    model.enable_infmae_late_lora = True
    model.infmae_direct_encoder = nn.Module()
    model.infmae_direct_encoder.base = nn.Linear(3, 3)
    model.infmae_direct_encoder.lora_A = nn.Linear(3, 2, bias=False)
    model.infmae_direct_encoder.lora_B = nn.Linear(2, 3, bias=False)
    model.infmae_direct_input_normalizer = nn.BatchNorm1d(3)
    model.infmae_direct_adapter = nn.Linear(3, 3)
    model.infmae_direct_cls_projection = nn.Linear(3, 3)

    checkpoint_state = model.state_dict()
    assert any(
        key.startswith("infmae_direct_encoder.lora_") for key in checkpoint_state
    )
    assert not any(
        key.startswith("infmae_direct_encoder.base.") for key in checkpoint_state
    )
    assert not any(
        key.startswith("infmae_direct_input_normalizer.") for key in checkpoint_state
    )


def test_a7_strict_frontend_tuning_freezes_original_grounding_trunk():
    fusion_module = _load_fusion_module()
    model = object.__new__(fusion_module.MMVGFusion)
    nn.Module.__init__(model)
    model.enable_infmae_strict_frontend_tuning = True
    model.enable_infmae_direct = True
    model.enable_infmae_hierarchical_adapter = True
    model.enable_infmae_late_lora = True
    model.enable_infmae_alignment = True

    # Representative RGB/trunk parameters must stay fixed.
    model.clip = nn.Linear(3, 3)
    model.cross_fusion_layers_vt = nn.ModuleList([nn.Linear(3, 3)])
    model.vl_transformer = nn.Linear(3, 3)
    model.bbox_embed = nn.Linear(3, 4)

    # The A7 thermal adapter contains both the hierarchical adapter and its
    # semantic bridge; both are intentionally part of the trainable frontend.
    model.infmae_direct_adapter = nn.Sequential(nn.Linear(3, 3), nn.Linear(3, 3))
    model.infmae_tir_text_projector = nn.Linear(3, 3)
    model.infmae_tir_rgb_projector = nn.Linear(3, 3)
    model.infmae_direct_encoder = nn.Module()
    model.infmae_direct_encoder.frozen_block = nn.Linear(3, 3)
    model.infmae_direct_encoder.late = nn.Module()
    model.infmae_direct_encoder.late.lora_A = nn.Linear(3, 2, bias=False)
    model.infmae_direct_encoder.late.lora_B = nn.Linear(2, 3, bias=False)

    model._configure_infmae_strict_frontend_tuning()

    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert trainable_names
    assert all(
        name.startswith(
            (
                'infmae_direct_adapter.',
                'infmae_tir_text_projector.',
                'infmae_tir_rgb_projector.',
            )
        )
        or (
            name.startswith('infmae_direct_encoder.')
            and ('.lora_A.' in name or '.lora_B.' in name)
        )
        for name in trainable_names
    )
    assert not any(name.startswith('clip.') for name in trainable_names)
    assert not any(name.startswith('cross_fusion_layers_vt.') for name in trainable_names)
    assert not any(name.startswith('vl_transformer.') for name in trainable_names)
    assert not any(name.startswith('bbox_embed.') for name in trainable_names)
    assert not any(
        name.startswith('infmae_direct_encoder.frozen_block.')
        for name in trainable_names
    )

    # The historical training loop calls ``set_HiLoRA`` again at later
    # milestones.  The strict alias records its mask so that method restores
    # the same trainability contract after any legacy scheduler action.
    assert model._strict_frontend_tuning_configured is True
    for parameter in model.clip.parameters():
        parameter.requires_grad_(True)
    model.set_HiLoRA(
        types.SimpleNamespace(open_lora=False, open_text_guided_fusion=False)
    )
    assert not any(parameter.requires_grad for parameter in model.clip.parameters())


def test_a7_f3_peft_preserves_converged_a7_interface():
    fusion_module = _load_fusion_module()
    model = object.__new__(fusion_module.MMVGFusion)
    nn.Module.__init__(model)
    model.enable_infmae_f3_peft_tuning = True
    model.enable_infmae_direct = True
    model.enable_infmae_hierarchical_adapter = True
    model.enable_infmae_late_lora = True
    model.enable_infmae_alignment = True

    model.clip = nn.Linear(3, 3)
    model.cross_fusion_layers_vt = nn.ModuleList([nn.Linear(3, 3)])
    model.vl_transformer = nn.Linear(3, 3)
    model.bbox_embed = nn.Linear(3, 4)
    model.infmae_direct_adapter = nn.Sequential(nn.Linear(3, 3), nn.LayerNorm(3))
    model.infmae_tir_text_projector = nn.Linear(3, 3)
    model.infmae_tir_rgb_projector = nn.Linear(3, 3)
    model.infmae_direct_encoder = nn.Module()
    model.infmae_direct_encoder.frozen_block = nn.Linear(3, 3)
    model.infmae_direct_encoder.late = nn.Module()
    model.infmae_direct_encoder.late.lora_A = nn.Linear(3, 2, bias=False)
    model.infmae_direct_encoder.late.lora_B = nn.Linear(2, 3, bias=False)

    model._configure_infmae_f3_peft_tuning()
    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert trainable_names
    assert all(
        name.startswith(('infmae_tir_text_projector.', 'infmae_tir_rgb_projector.'))
        or (
            name.startswith('infmae_direct_encoder.')
            and ('.lora_A.' in name or '.lora_B.' in name)
        )
        for name in trainable_names
    )
    assert not any(name.startswith('infmae_direct_adapter.') for name in trainable_names)
    assert not any(name.startswith('clip.') for name in trainable_names)
    assert not any(name.startswith('cross_fusion_layers_vt.') for name in trainable_names)
    assert not any(name.startswith('vl_transformer.') for name in trainable_names)
    assert not any(name.startswith('bbox_embed.') for name in trainable_names)

    assert model._f3_peft_tuning_configured is True
    for parameter in model.clip.parameters():
        parameter.requires_grad_(True)
    model.set_HiLoRA(
        types.SimpleNamespace(open_lora=False, open_text_guided_fusion=False)
    )
    assert not any(parameter.requires_grad for parameter in model.clip.parameters())


def test_a8_evidence_peft_only_opens_local_transfer_and_alignment_heads():
    fusion_module = _load_fusion_module()
    model = object.__new__(fusion_module.MMVGFusion)
    nn.Module.__init__(model)
    model.enable_infmae_local_evidence_adapter = True
    model.enable_infmae_direct = True
    model.enable_infmae_hierarchical_adapter = True
    model.enable_infmae_alignment = True

    model.clip = nn.Linear(3, 3)
    model.cross_fusion_layers_vt = nn.ModuleList([nn.Linear(3, 3)])
    model.vl_transformer = nn.Linear(3, 3)
    model.bbox_embed = nn.Linear(3, 4)
    model.infmae_direct_adapter = nn.Sequential(nn.Linear(3, 3), nn.LayerNorm(3))
    model.infmae_direct_encoder = nn.Linear(3, 3)
    model.infmae_local_evidence_adapter = nn.Sequential(nn.Linear(3, 3), nn.Linear(3, 3))
    model.infmae_tir_text_projector = nn.Linear(3, 3)
    model.infmae_tir_rgb_projector = nn.Linear(3, 3)

    model._configure_infmae_a8_evidence_peft_tuning()
    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert trainable_names
    assert all(
        name.startswith(
            (
                'infmae_local_evidence_adapter.',
                'infmae_tir_text_projector.',
                'infmae_tir_rgb_projector.',
            )
        )
        for name in trainable_names
    )
    assert not any(name.startswith('infmae_direct_adapter.') for name in trainable_names)
    assert not any(name.startswith('infmae_direct_encoder.') for name in trainable_names)
    assert not any(name.startswith('clip.') for name in trainable_names)
    assert not any(name.startswith('cross_fusion_layers_vt.') for name in trainable_names)
    assert not any(name.startswith('vl_transformer.') for name in trainable_names)
    assert not any(name.startswith('bbox_embed.') for name in trainable_names)

    assert model._a8_evidence_peft_tuning_configured is True
    for parameter in model.clip.parameters():
        parameter.requires_grad_(True)
    model.set_HiLoRA(
        types.SimpleNamespace(open_lora=False, open_text_guided_fusion=False)
    )
    assert not any(parameter.requires_grad for parameter in model.clip.parameters())


def test_tcrc_peft_fixes_a5_frontend_and_opens_only_reliability_calibration():
    fusion_module = _load_fusion_module()
    model = object.__new__(fusion_module.MMVGFusion)
    nn.Module.__init__(model)
    model.enable_infmae_text_reliability_calibration = True
    model.enable_infmae_direct = True
    model.enable_infmae_semantic_bridge = True
    model.enable_infmae_late_lora = True
    model.enable_infmae_alignment = True

    model.clip = nn.Linear(3, 3)
    model.cross_fusion_layers_vt = nn.ModuleList([nn.Linear(3, 3)])
    model.vl_transformer = nn.Linear(3, 3)
    model.bbox_embed = nn.Linear(3, 4)
    model.infmae_direct_adapter = nn.Sequential(nn.Linear(3, 3), nn.LayerNorm(3))
    model.infmae_direct_encoder = nn.Linear(3, 3)
    model.infmae_tir_text_projector = nn.Linear(3, 3)
    model.infmae_tir_rgb_projector = nn.Linear(3, 3)
    model.text_conditioned_reliability_calibrator = nn.Sequential(
        nn.Linear(3, 3), nn.Linear(3, 1)
    )

    model._configure_infmae_tcrc_peft_tuning()
    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert trainable_names
    assert all(
        name.startswith('text_conditioned_reliability_calibrator.')
        for name in trainable_names
    )
    assert not any(name.startswith('infmae_direct_adapter.') for name in trainable_names)
    assert not any(name.startswith('infmae_direct_encoder.') for name in trainable_names)
    assert not any(name.startswith('infmae_tir_') for name in trainable_names)
    assert not any(name.startswith('clip.') for name in trainable_names)
    assert not any(name.startswith('cross_fusion_layers_vt.') for name in trainable_names)
    assert not any(name.startswith('vl_transformer.') for name in trainable_names)
    assert not any(name.startswith('bbox_embed.') for name in trainable_names)

    assert model._tcrc_peft_tuning_configured is True
    for parameter in model.clip.parameters():
        parameter.requires_grad_(True)
    model.set_HiLoRA(
        types.SimpleNamespace(open_lora=False, open_text_guided_fusion=False)
    )
    assert not any(parameter.requires_grad for parameter in model.clip.parameters())


def test_a5_f2_peft_opens_only_thermal_lora_and_alignment_heads():
    fusion_module = _load_fusion_module()
    model = object.__new__(fusion_module.MMVGFusion)
    nn.Module.__init__(model)
    model.enable_infmae_f2_peft_tuning = True
    model.enable_infmae_direct = True
    model.enable_infmae_semantic_bridge = True
    model.enable_infmae_late_lora = True
    model.enable_infmae_f2_lora = True
    model.enable_infmae_alignment = True

    model.clip = nn.Linear(3, 3)
    model.cross_fusion_layers_vt = nn.ModuleList([nn.Linear(3, 3)])
    model.vl_transformer = nn.Linear(3, 3)
    model.bbox_embed = nn.Linear(3, 4)
    model.infmae_direct_adapter = nn.Sequential(nn.Linear(3, 3), nn.LayerNorm(3))
    model.infmae_tir_text_projector = nn.Linear(3, 3)
    model.infmae_tir_rgb_projector = nn.Linear(3, 3)
    model.infmae_direct_encoder = nn.Module()
    model.infmae_direct_encoder.has_f2_lora = True
    model.infmae_direct_encoder.has_late_lora = True
    model.infmae_direct_encoder.frozen_block = nn.Linear(3, 3)
    model.infmae_direct_encoder.blocks2 = nn.ModuleList([nn.Module()])
    model.infmae_direct_encoder.blocks2[0].adapt = nn.Module()
    model.infmae_direct_encoder.blocks2[0].adapt.lora_A = nn.Linear(3, 2, bias=False)
    model.infmae_direct_encoder.blocks2[0].adapt.lora_B = nn.Linear(2, 3, bias=False)
    model.infmae_direct_encoder.blocks3 = nn.ModuleList([nn.Module()])
    model.infmae_direct_encoder.blocks3[0].adapt = nn.Module()
    model.infmae_direct_encoder.blocks3[0].adapt.lora_A = nn.Linear(3, 2, bias=False)
    model.infmae_direct_encoder.blocks3[0].adapt.lora_B = nn.Linear(2, 3, bias=False)

    model._configure_infmae_f2_peft_tuning()
    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    assert trainable_names
    assert all(
        name.startswith(('infmae_tir_text_projector.', 'infmae_tir_rgb_projector.'))
        or (
            name.startswith('infmae_direct_encoder.')
            and ('.lora_A.' in name or '.lora_B.' in name)
        )
        for name in trainable_names
    )
    assert any('.blocks2.' in name for name in trainable_names)
    assert any('.blocks3.' in name for name in trainable_names)
    assert not any(name.startswith('infmae_direct_adapter.') for name in trainable_names)
    assert not any(name.startswith('infmae_direct_encoder.frozen_block.') for name in trainable_names)
    assert not any(name.startswith('clip.') for name in trainable_names)
    assert not any(name.startswith('cross_fusion_layers_vt.') for name in trainable_names)
    assert not any(name.startswith('vl_transformer.') for name in trainable_names)
    assert not any(name.startswith('bbox_embed.') for name in trainable_names)

    assert model._f2_peft_tuning_configured is True
    for parameter in model.clip.parameters():
        parameter.requires_grad_(True)
    model.set_HiLoRA(
        types.SimpleNamespace(open_lora=False, open_text_guided_fusion=False)
    )
    assert not any(parameter.requires_grad for parameter in model.clip.parameters())
