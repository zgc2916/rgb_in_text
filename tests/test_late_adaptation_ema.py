from types import SimpleNamespace

import pytest
import torch
from torch import nn

from models.late_adaptation import parameter_group, LateAdaptationGrounder
from utils.parameter_ema import ParameterEMA
from script_train.clip_late_adaptation_control import runtime as r


@pytest.mark.parametrize('modality', ['lora_rgb', 'lora_ir'])
def test_late_visual_adapter_scope_preserves_language_and_early_visual_layers(modality):
    def name(branch, layer):
        return f'base.clip.base_model.model.{branch}.encoder.layers.{layer}.self_attn.q_proj.lora_A.{modality}.weight'
    assert parameter_group(name('vision_model', 8)) == 'visual_lora'
    assert parameter_group(name('vision_model', 11)) == 'visual_lora'
    assert parameter_group(name('vision_model', 7)) is None
    assert parameter_group(name('text_model', 8)) is None
    assert parameter_group(name('vision_model', 8).replace('.lora_A.' + modality, '.base_layer')) is None


def test_fusion_and_decoder_scope_are_explicit():
    assert parameter_group('base.iafv3_alpha') == 'fusion'
    assert parameter_group('base.cross_fusion_layers_vt.0.q_proj.weight') == 'fusion'
    assert parameter_group('base.vl_transformer.encoder.layers.3.self_attn.in_proj_weight') == 'decoder'
    assert parameter_group('base.vl_transformer.encoder.layers.2.self_attn.in_proj_weight') is None
    assert parameter_group('base.seg_conv1.weight') is None


def test_optimizer_hook_updates_ema_per_optimizer_step_and_restores_raw_after_exception():
    model = nn.Linear(1, 1, bias=False)
    ema = ParameterEMA(model, ['weight'])
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    handle = optimizer.register_step_post_hook(lambda *_: ema.update(model))
    for _ in range(2):
        optimizer.zero_grad()
        for _ in range(3):
            (model(torch.ones(1, 1)).sum() / 3).backward()
        optimizer.step()
    assert ema.updates == 2
    raw = model.weight.detach().clone()
    with pytest.raises(RuntimeError, match='probe'):
        with ema.apply(model):
            assert not torch.equal(model.weight, raw)
            raise RuntimeError('probe')
    assert torch.equal(model.weight, raw)
    handle.remove()


def test_ema_recovery_matches_uninterrupted_next_update():
    model = nn.Linear(1, 1, bias=False)
    ema = ParameterEMA(model, ['weight'])
    with torch.no_grad():
        model.weight.add_(1.)
    ema.update(model)
    state = ema.state_dict()
    restored = ParameterEMA(model, ['weight'])
    restored.load_state_dict(state)
    with torch.no_grad():
        model.weight.sub_(.5)
    ema.update(model)
    restored.update(model)
    assert ema.updates == restored.updates
    assert torch.equal(ema.shadow['weight'], restored.shadow['weight'])


def test_ema_rejects_different_scope():
    model = nn.Linear(1, 1)
    a, b = ParameterEMA(model, ['weight']), ParameterEMA(model, ['bias'])
    with pytest.raises(ValueError, match='scope'):
        b.load_state_dict(a.state_dict())


def test_gt_never_enters_the_single_model_forward():
    model = object.__new__(LateAdaptationGrounder)
    nn.Module.__init__(model)
    with pytest.raises(ValueError, match='GT boxes'):
        model(None, None, target_boxes=torch.ones(1, 4))


def test_three_seeds_use_paired_original_checkpoints_and_equal_training_budget(tmp_path):
    recipes = [r.recipe(tmp_path, seed=seed) for seed in r.SEEDS]
    assert {a.seed for a in recipes} == {13, 14, 15}
    assert len({a.exchange_base_sha256 for a in recipes}) == 3
    assert all(a.epochs == 30 and a.batch_size*a.grad_accum_steps*a.world_size == 96 for a in recipes)
    assert all(not a.relative_geometry_enabled and a.target_scale_floor_pixels == 0 for a in recipes)


def test_validation_gate_cannot_be_passed_by_one_good_seed():
    runs = {}
    for seed in r.SEEDS:
        baseline = dict(r.reference(seed)['validation'])
        if seed == 14:
            baseline['accu'] += .02
            baseline['accu_small'] += .03
        runs[f'late_adaptation_seed{seed}'] = baseline
    assert not r.choose_promotion(runs)['passed']


def test_gradient_audit_handles_none_hooks_without_inventing_updates():
    from script_train.clip_late_adaptation_control.preflight import gradient_l1
    assert gradient_l1(None) == 0.
    assert gradient_l1(torch.tensor([1., -2.])) == 3.
