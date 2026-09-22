import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_tsar_modules():
    spec = importlib.util.spec_from_file_location(
        "tsar_modules_for_test",
        REPO_ROOT / "models" / "tsar_modules.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_train_parser():
    """Load the CLI parser without importing optional model backbones."""
    models_stub = type(sys)("models")
    models_stub.build_model = lambda args: None
    datasets_stub = type(sys)("datasets")
    datasets_stub.build_dataset = lambda split, args: None
    engine_stub = type(sys)("engine")
    engine_stub.train_one_epoch = lambda *args, **kwargs: None
    engine_stub.validate = lambda *args, **kwargs: None
    utils_stub = types.ModuleType("utils")
    utils_stub.__path__ = []
    utils_misc_stub = types.ModuleType("utils.misc")
    utils_stub.misc = utils_misc_stub
    original_modules = {
        name: sys.modules.get(name)
        for name in ("models", "datasets", "engine", "utils", "utils.misc")
    }
    sys.modules.update({
        "models": models_stub,
        "datasets": datasets_stub,
        "engine": engine_stub,
        "utils": utils_stub,
        "utils.misc": utils_misc_stub,
    })
    try:
        spec = importlib.util.spec_from_file_location(
            "mmvg_train_for_test",
            REPO_ROOT / "train_val" / "mmvg_train.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, original_module in original_modules.items():
            if original_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original_module


def test_gqr_cli_is_explicitly_opt_in():
    parser = _load_train_parser().get_args_parser()
    defaults = parser.parse_args([])
    assert defaults.enable_gqr is False
    assert defaults.gqr_tau == 0.25
    assert defaults.gqr_eta_max == 2.0
    assert defaults.gqr_teacher_margin == 0.0
    assert defaults.grad_accum_steps == 1
    assert defaults.infmae_grounding_adapter_gradient_scale == 1.0

    enabled = parser.parse_args(["--FusionMethod", "GQRv1", "--enable_gqr"])
    assert enabled.enable_gqr is True
    assert enabled.FusionMethod == "GQRv1"


def test_text_conditioned_heads_and_router_shapes_and_backward():
    modules = _load_tsar_modules()
    torch.manual_seed(7)
    batch_size, patch_count, dim = 2, 196, 32
    rgb_tokens = torch.randn(batch_size, patch_count + 1, dim, requires_grad=True)
    tir_tokens = torch.randn(batch_size, patch_count + 1, dim, requires_grad=True)
    text_embed = torch.randn(batch_size, dim, requires_grad=True)

    rgb_head = modules.TextConditionedModalityHead(dim)
    tir_head = modules.TextConditionedModalityHead(dim)
    router = modules.GroundingQualityRouter(dim)

    rgb_aux = rgb_head(rgb_tokens, text_embed)
    tir_aux = tir_head(tir_tokens, text_embed)
    router_logits = router(rgb_aux["target_feat"], tir_aux["target_feat"], text_embed)

    assert rgb_aux["box"].shape == (batch_size, 4)
    assert tir_aux["box"].shape == (batch_size, 4)
    assert rgb_aux["attn"].shape == (batch_size, patch_count)
    assert torch.allclose(rgb_aux["attn"].sum(dim=-1), torch.ones(batch_size), atol=1e-6)
    assert router_logits.shape == (batch_size, 2)
    assert torch.allclose(
        router_logits.softmax(dim=-1).sum(dim=-1),
        torch.ones(batch_size),
        atol=1e-6,
    )

    loss = rgb_aux["box"].sum() + tir_aux["box"].sum() + router_logits.square().mean()
    loss.backward()
    assert all(parameter.grad is not None for parameter in router.parameters())


def test_text_conditioned_reliability_calibrator_is_identity_then_learns():
    modules = _load_tsar_modules()
    torch.manual_seed(19)
    batch_size, patch_count, dim = 2, 196, 32
    calibrator = modules.TextConditionedReliabilityCalibrator(
        dim,
        bottleneck_dim=8,
        max_logit_shift=0.75,
    )
    rgb = torch.randn(batch_size, patch_count, dim, requires_grad=True)
    tir = torch.randn(batch_size, patch_count, dim, requires_grad=True)
    text = torch.randn(batch_size, dim, requires_grad=True)
    base_weight = torch.sigmoid(
        torch.randn(batch_size, patch_count, 1)
    ).detach().requires_grad_()

    calibrated, aux = calibrator(rgb, tir, text, base_weight)
    # A zero final head leaves the established IAFv3 fusion path bitwise
    # unchanged while still letting that head acquire a first-step gradient.
    assert torch.equal(calibrated, base_weight)
    assert calibrated.shape == base_weight.shape
    assert all(torch.isfinite(value).item() for value in aux.values())
    calibrated.square().mean().backward()
    assert calibrator.correction_head.weight.grad is not None
    assert torch.count_nonzero(calibrator.correction_head.weight.grad).item() > 0
    assert calibrator.visual_projection.weight.grad is not None
    assert torch.count_nonzero(calibrator.visual_projection.weight.grad).item() == 0

    calibrator.zero_grad(set_to_none=True)
    with torch.no_grad():
        calibrator.correction_head.weight.normal_(mean=0.0, std=0.02)
    calibrated, _ = calibrator(rgb, tir, text, base_weight)
    assert torch.all(calibrated >= 0.0)
    assert torch.all(calibrated <= 1.0)
    calibrated.square().mean().backward()
    assert calibrator.visual_projection.weight.grad is not None
    assert calibrator.text_projection.weight.grad is not None


def test_gqr_losses_are_finite_and_teacher_does_not_backpropagate_to_boxes():
    from utils.loss_utils import gqr_ramp_scale, trans_vg_loss

    torch.manual_seed(11)
    batch_size = 2
    args = SimpleNamespace(
        use_contrastive_loss=False,
        use_rtcc_constrain_loss=False,
        use_mask_loss=False,
        enable_gqr=True,
        gqr_tau=0.25,
        gqr_aux_weight=0.25,
        gqr_router_weight=0.20,
        gqr_start_epoch=5,
        gqr_ramp_epochs=5,
    )
    pred_box = torch.sigmoid(torch.randn(batch_size, 4)).detach().requires_grad_()
    target = torch.tensor(
        [[0.50, 0.50, 0.30, 0.25], [0.35, 0.55, 0.20, 0.30]],
        dtype=torch.float32,
    )
    rgb_box = torch.sigmoid(torch.randn(batch_size, 4)).detach().requires_grad_()
    tir_box = torch.sigmoid(torch.randn(batch_size, 4)).detach().requires_grad_()
    router_logits = torch.randn(batch_size, 2, requires_grad=True)
    aux = {
        "rgb_aux_box": rgb_box,
        "tir_aux_box": tir_box,
        "router_logits": router_logits,
        "router_prob": router_logits.softmax(dim=-1),
        "gqr_eta": torch.tensor(0.0),
        "w_rgb_base_mean": torch.tensor(0.5),
        "w_rgb_final_mean": torch.tensor(0.5),
    }

    losses = trans_vg_loss(
        args,
        pred_box,
        target,
        tgt_mask=None,
        text_eos=None,
        aux=aux,
        epoch=5,
    )
    optimizable = [value for key, value in losses.items() if key.startswith("loss_")]
    assert optimizable
    assert all(torch.isfinite(value).item() for value in optimizable)
    assert gqr_ramp_scale(args, 4) == 0.0
    assert abs(gqr_ramp_scale(args, 5) - 0.2) < 1e-6
    assert abs(losses["gqr_ramp_scale"].item() - 0.2) < 1e-6

    router_grad_to_rgb_box = torch.autograd.grad(
        losses["loss_gqr_router"],
        rgb_box,
        allow_unused=True,
        retain_graph=True,
    )[0]
    assert router_grad_to_rgb_box is None

    sum(optimizable).backward()
    assert rgb_box.grad is not None
    assert tir_box.grad is not None
    assert router_logits.grad is not None


def test_gqr_teacher_margin_excludes_ambiguous_router_labels():
    from utils.loss_utils import trans_vg_loss

    args = SimpleNamespace(
        use_contrastive_loss=False,
        use_rtcc_constrain_loss=False,
        use_mask_loss=False,
        enable_gqr=True,
        gqr_tau=0.25,
        gqr_aux_weight=0.0,
        gqr_router_weight=1.0,
        gqr_start_epoch=0,
        gqr_ramp_epochs=1,
        gqr_teacher_margin=0.05,
    )
    target = torch.tensor(
        [[0.50, 0.50, 0.20, 0.20], [0.50, 0.50, 0.20, 0.20]],
        dtype=torch.float32,
    )
    pred_box = target.clone().requires_grad_()
    # The first pair has equal IoU (ambiguous); the second pair has a clear
    # RGB advantage and must remain in the router KL supervision.
    rgb_box = target.clone().requires_grad_()
    tir_box = torch.tensor(
        [[0.50, 0.50, 0.20, 0.20], [0.50, 0.50, 0.10, 0.10]],
        dtype=torch.float32,
        requires_grad=True,
    )
    router_logits = torch.tensor(
        [[0.3, -0.3], [-0.3, 0.3]], dtype=torch.float32, requires_grad=True
    )
    losses = trans_vg_loss(
        args,
        pred_box,
        target,
        tgt_mask=None,
        text_eos=None,
        aux={
            'rgb_aux_box': rgb_box,
            'tir_aux_box': tir_box,
            'router_logits': router_logits,
        },
        epoch=0,
    )
    assert torch.isfinite(losses['loss_gqr_router']).item()
    assert torch.allclose(losses['router_teacher_coverage'], torch.tensor(0.5))

    router_grad = torch.autograd.grad(losses['loss_gqr_router'], router_logits)[0]
    assert torch.allclose(router_grad[0], torch.zeros_like(router_grad[0]))
    assert not torch.allclose(router_grad[1], torch.zeros_like(router_grad[1]))


def test_infmae_a5_target_losses_are_finite_and_only_update_tir_embeddings():
    from utils.loss_utils import (
        infmae_alignment_adapter_ramp_scale,
        infmae_alignment_ramp_scale,
        infmae_rgb_tir_ramp_scale,
        trans_vg_loss,
    )

    torch.manual_seed(41)
    batch_size, dim = 3, 8
    args = SimpleNamespace(
        use_contrastive_loss=False,
        use_rtcc_constrain_loss=False,
        use_mask_loss=False,
        enable_gqr=False,
        infmae_alignment_mode='both',
        infmae_tir_text_weight=0.10,
        infmae_rgb_tir_weight=0.05,
        infmae_tir_text_tau=0.07,
        infmae_alignment_start_epoch=0,
        infmae_alignment_ramp_epochs=5,
        infmae_rgb_tir_start_epoch=3,
        infmae_rgb_tir_ramp_epochs=2,
        infmae_alignment_adapter_start_epoch=4,
        infmae_alignment_adapter_ramp_epochs=2,
    )
    pred_box = torch.sigmoid(torch.randn(batch_size, 4)).detach().requires_grad_()
    target = torch.tensor(
        [[0.50, 0.50, 0.30, 0.25], [0.35, 0.55, 0.20, 0.30], [0.65, 0.35, 0.18, 0.20]],
        dtype=torch.float32,
    )
    tir_text = torch.nn.functional.normalize(
        torch.randn(batch_size, dim), dim=-1
    ).detach().requires_grad_()
    text_teacher = torch.nn.functional.normalize(
        torch.randn(batch_size, dim), dim=-1
    ).detach().requires_grad_()
    tir_rgb = torch.nn.functional.normalize(
        torch.randn(batch_size, dim), dim=-1
    ).detach().requires_grad_()
    rgb_teacher = torch.nn.functional.normalize(
        torch.randn(batch_size, dim), dim=-1
    ).detach().requires_grad_()

    losses = trans_vg_loss(
        args,
        pred_box,
        target,
        tgt_mask=None,
        text_eos=None,
        aux={
            'tir_text_embedding': tir_text,
            'text_embedding': text_teacher,
            'tir_rgb_embedding': tir_rgb,
            'rgb_embedding': rgb_teacher,
        },
        epoch=3,
    )
    assert torch.isfinite(losses['loss_infmae_tir_text']).item()
    assert torch.isfinite(losses['loss_infmae_rgb_tir']).item()
    assert losses['tir_text_candidate_count'].item() == batch_size
    assert abs(infmae_alignment_ramp_scale(args, 3) - 0.8) < 1e-6
    assert abs(infmae_rgb_tir_ramp_scale(args, 2)) < 1e-6
    assert abs(infmae_rgb_tir_ramp_scale(args, 3) - 0.5) < 1e-6
    assert abs(losses['infmae_rgb_tir_ramp_scale'].item() - 0.5) < 1e-6
    assert abs(infmae_alignment_adapter_ramp_scale(args, 3)) < 1e-6
    assert abs(infmae_alignment_adapter_ramp_scale(args, 4) - 0.5) < 1e-6
    assert abs(losses['infmae_alignment_adapter_scale'].item()) < 1e-6

    sum(value for key, value in losses.items() if key.startswith('loss_')).backward()
    assert tir_text.grad is not None
    assert tir_rgb.grad is not None
    assert text_teacher.grad is None
    assert rgb_teacher.grad is None


def test_infmae_spatial_attention_loss_is_finite_and_updates_attention_only():
    from utils.loss_utils import trans_vg_loss

    torch.manual_seed(43)
    batch_size, dim, patch_count, stages = 3, 8, 16, 4
    args = SimpleNamespace(
        FusionMethod='InfMAEA6SpatialFT',
        use_contrastive_loss=False,
        use_rtcc_constrain_loss=False,
        use_mask_loss=False,
        enable_gqr=False,
        infmae_alignment_mode='both',
        infmae_tir_text_weight=0.10,
        infmae_rgb_tir_weight=0.05,
        infmae_tir_text_tau=0.07,
        infmae_alignment_start_epoch=0,
        infmae_alignment_ramp_epochs=5,
        infmae_rgb_tir_start_epoch=0,
        infmae_rgb_tir_ramp_epochs=1,
        infmae_alignment_adapter_start_epoch=0,
        infmae_alignment_adapter_ramp_epochs=1,
    )
    pred_box = torch.sigmoid(torch.randn(batch_size, 4)).detach().requires_grad_()
    target = torch.tensor(
        [[0.50, 0.50, 0.30, 0.25], [0.35, 0.55, 0.20, 0.30], [0.65, 0.35, 0.18, 0.20]],
        dtype=torch.float32,
    )
    tir_text = torch.nn.functional.normalize(torch.randn(batch_size, dim), dim=-1).detach().requires_grad_()
    text_teacher = torch.nn.functional.normalize(torch.randn(batch_size, dim), dim=-1).detach().requires_grad_()
    tir_rgb = torch.nn.functional.normalize(torch.randn(batch_size, dim), dim=-1).detach().requires_grad_()
    rgb_teacher = torch.nn.functional.normalize(torch.randn(batch_size, dim), dim=-1).detach().requires_grad_()
    attention_logits = torch.randn(batch_size, stages, patch_count, requires_grad=True)
    attention = attention_logits.softmax(dim=-1)
    target_distribution = torch.softmax(torch.randn(batch_size, patch_count), dim=-1)

    losses = trans_vg_loss(
        args,
        pred_box,
        target,
        tgt_mask=None,
        text_eos=None,
        aux={
            'tir_text_embedding': tir_text,
            'text_embedding': text_teacher,
            'tir_rgb_embedding': tir_rgb,
            'rgb_embedding': rgb_teacher,
            'infmae_spatial_attention': attention,
            'infmae_spatial_target_distribution': target_distribution,
        },
        epoch=4,
    )
    assert torch.isfinite(losses['loss_infmae_spatial']).item()
    assert 0.0 <= losses['infmae_spatial_target_mass'].item() <= 1.0
    sum(value for key, value in losses.items() if key.startswith('loss_')).backward()
    assert attention_logits.grad is not None
    assert torch.count_nonzero(attention_logits.grad).item() > 0
    assert text_teacher.grad is None
    assert rgb_teacher.grad is None


def test_infmae_spatial_mass_loss_keeps_discriminative_in_box_attention_free():
    from utils.loss_utils import (
        _infmae_spatial_attention_config,
        _infmae_spatial_attention_mass_loss,
        infmae_spatial_ramp_scale,
    )

    torch.manual_seed(47)
    logits = torch.randn(2, 4, 16, requires_grad=True)
    attention = logits.softmax(dim=-1)
    target_distribution = torch.zeros(2, 16)
    target_distribution[:, 5] = 0.8
    target_distribution[:, 6] = 0.2
    loss, target_mass = _infmae_spatial_attention_mass_loss(
        attention,
        target_distribution,
    )
    assert torch.isfinite(loss).item()
    assert 0.0 <= target_mass.item() <= 1.0
    loss.backward()
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad).item() > 0

    args = SimpleNamespace(FusionMethod='InfMAEA6MassFT')
    assert _infmae_spatial_attention_config(args) == ('mass', 0.005, 50, 10)
    assert infmae_spatial_ramp_scale(args, 49) == 0.0
    assert abs(infmae_spatial_ramp_scale(args, 50) - 0.1) < 1e-6
    assert infmae_spatial_ramp_scale(args, 59) == 1.0


def test_infmae_a3_alias_resolves_to_text_alignment_only():
    parser = _load_train_parser().get_args_parser()
    args = parser.parse_args(['--FusionMethod', 'InfMAEA3'])
    assert args.infmae_alignment_mode == 'auto'
    assert args.infmae_alignment_adapter_only is False

    from utils.loss_utils import _infmae_alignment_mode
    assert _infmae_alignment_mode(args) == 'tir_text'


def test_infmae_a5_bridge_alias_resolves_to_both_target_losses():
    parser = _load_train_parser().get_args_parser()
    args = parser.parse_args(['--FusionMethod', 'InfMAEA5Bridge'])

    from utils.loss_utils import _infmae_alignment_mode
    assert _infmae_alignment_mode(args) == 'both'


def test_infmae_a5_bridge_ft_alias_resolves_to_both_target_losses():
    parser = _load_train_parser().get_args_parser()
    args = parser.parse_args(['--FusionMethod', 'InfMAEA5BridgeFT'])

    from utils.loss_utils import _infmae_alignment_mode
    assert _infmae_alignment_mode(args) == 'both'


def test_infmae_a6_spatial_ft_alias_resolves_to_both_target_losses():
    parser = _load_train_parser().get_args_parser()
    args = parser.parse_args(['--FusionMethod', 'InfMAEA6SpatialFT'])

    from utils.loss_utils import _infmae_alignment_mode
    assert _infmae_alignment_mode(args) == 'both'


def test_infmae_a6_mass_ft_alias_resolves_to_both_target_losses():
    parser = _load_train_parser().get_args_parser()
    args = parser.parse_args(['--FusionMethod', 'InfMAEA6MassFT'])

    from utils.loss_utils import _infmae_alignment_mode
    assert _infmae_alignment_mode(args) == 'both'


def test_infmae_a7_hadapter_aliases_resolve_to_both_target_losses():
    parser = _load_train_parser().get_args_parser()

    from utils.loss_utils import _infmae_alignment_mode

    for method in (
        'InfMAEA7HAdapter',
        'InfMAEA7HAdapterFT',
        'InfMAEA7HAdapterFTStrict',
        'InfMAEA7HAdapterF3PEFT',
        'InfMAEA8EvidencePEFT',
        'InfMAEA5BridgeFTTCRC',
        'InfMAEA5BridgeFTF2',
        'InfMAEA5BridgeFTF2PEFT',
    ):
        args = parser.parse_args(['--FusionMethod', method])
        assert _infmae_alignment_mode(args) == 'both'


def test_mmvg_gradient_accumulation_updates_once_per_group():
    from engine import train_one_epoch

    class ToyMMVG(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.box_logits = torch.nn.Parameter(torch.zeros(4))

        def forward(self, image, _text):
            batch_size = image.shape[0]
            box = self.box_logits.sigmoid().unsqueeze(0).expand(batch_size, -1)
            return box, None, None, None, None

    class CountingSGD(torch.optim.SGD):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.step_count = 0

        def step(self, closure=None):
            self.step_count += 1
            return super().step(closure=closure)

    args = SimpleNamespace(
        model_name='MMVGFusion',
        old_dataloader=False,
        use_contrastive_loss=False,
        use_rtcc_constrain_loss=False,
        use_mask_loss=False,
        enable_gqr=False,
        grad_accum_steps=2,
    )
    batch = (
        torch.ones(1, 1),
        None,
        torch.tensor([[0.50, 0.50, 0.20, 0.20]]),
        torch.zeros(1, 1, 1),
    )
    model = ToyMMVG()
    optimizer = CountingSGD(model.parameters(), lr=0.01)
    train_one_epoch(
        args,
        model,
        [batch, batch, batch, batch, batch],
        optimizer,
        torch.device('cpu'),
        epoch=0,
        max_norm=1.0,
    )

    # Five micro-batches with accumulation 2 form groups of 2, 2, and 1.
    assert optimizer.step_count == 3
