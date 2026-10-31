import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parents[1]))
from models.region_anchor_exchange import RegionAnchorExchange, MODES, masked_read
from script_train.clip_anchor_control import runtime
from test_iafv3_equivalence import _load_fusion_module

torch.set_num_threads(2)


def fixture():
    torch.manual_seed(13)
    module = RegionAnchorExchange(4, 4, 8, 16, 12, 2)
    rgb, tir = [torch.randn(2, 17, 16) for _ in range(2)], [torch.randn(2, 17, 16) for _ in range(2)]
    base_r, base_t = [x.clone() for x in rgb], [x.clone() for x in tir]
    text, mask = torch.randn(2, 6, 12), torch.zeros(2, 6, dtype=torch.bool)
    mask[:, -2:] = True
    padding = torch.zeros(2, 16, dtype=torch.bool)
    padding[1, -4:] = True
    return module, [rgb, tir, base_r, base_t, text, mask, padding]


def activate(module):
    nn.init.normal_(module.rgb_output.weight, std=.1)
    nn.init.normal_(module.tir_output.weight, std=.1)
    nn.init.constant_(module.rgb_output.bias, .2)
    nn.init.constant_(module.tir_output.bias, -.2)


@pytest.mark.parametrize('mode', MODES)
def test_initialization_is_exact_c1_and_preserves_rng_integration(mode):
    module, data = fixture()
    result = module(*data, mode=mode)
    assert all(torch.equal(a, b) for outputs, bases in zip(result, data[2:4]) for a, b in zip(outputs, bases))
    assert sum(p.numel() for p in RegionAnchorExchange().parameters()) < 1_000_000


@pytest.mark.parametrize('mode', MODES)
def test_padding_cannot_change_new_evidence_and_cls_is_unchanged(mode):
    module, data = fixture()
    activate(module)
    a = module(*data, mode=mode)
    changed = copy.deepcopy(data)
    changed[4][changed[5]] = 10000.
    for levels in changed[:2]:
        for level in levels:
            level[:, 1:][changed[6]] = -10000.
    b = module(*changed, mode=mode)
    for direction, base in zip(range(2), data[2:4]):
        for i, (x, y) in enumerate(zip(a[direction], b[direction])):
            assert torch.equal(x, y)
            assert torch.equal(x[:, 0], base[i][:, 0])
            assert torch.equal(x[:, 1:][data[6]], base[i][:, 1:][data[6]])
    data[5].fill_(True)
    data[6].fill_(True)
    out = module(*data, mode=mode)
    assert all(torch.equal(a, b) for outputs, bases in zip(out, data[2:4]) for a, b in zip(outputs, bases))


def test_visual_control_ignores_text_and_separate_memory_changes_both_directions():
    module, data = fixture()
    activate(module)
    changed = copy.deepcopy(data)
    changed[4].neg_()
    def compare(mode):
        a, b = module(*data, mode=mode), module(*changed, mode=mode)
        return any(not torch.equal(x, y) for xs, ys in zip(a, b) for x, y in zip(xs, ys))
    assert not compare('anchor_visual')
    assert compare('anchor_text_shared') and compare('anchor_text_separate')
    a = module(*data, mode='anchor_text_shared')
    b = module(*data, mode='anchor_text_separate')
    assert all(any(not torch.equal(x, y) for x, y in zip(xs, ys)) for xs, ys in zip(a, b))


def test_both_outputs_and_full_text_reader_receive_finite_gradients_after_warm_start():
    module, data = fixture()
    optimizer = torch.optim.AdamW(module.parameters(), lr=.01)
    targets = [[torch.randn_like(x) for x in levels] for levels in data[2:4]]
    for step in range(2):
        optimizer.zero_grad()
        out = module(*data)
        sum((a - b).square().mean() for xs, ys in zip(out, targets) for a, b in zip(xs, ys)).backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())
        for p in (module.rgb_output.weight, module.tir_output.weight):
            assert p.grad.abs().sum() > 0
        if step:
            assert module.text_proj.weight.grad.abs().sum() > 0
            assert module.anchors.grad.abs().sum() > 0
        optimizer.step()


def test_empty_attention_and_invalid_configuration():
    x = torch.randn(2, 4, 8)
    assert torch.equal(masked_read(x, x, x, torch.ones(2, 4, dtype=torch.bool)), torch.zeros_like(x))
    for count in (0, 3, 15):
        with pytest.raises(ValueError):
            RegionAnchorExchange(anchor_count=count)


@pytest.mark.parametrize('variant', runtime.VARIANTS)
def test_actual_model_tuning_and_peft_switch_preserve_the_registered_scope(tmp_path, variant):
    fusion, _ = _load_fusion_module()
    model = object.__new__(fusion.MMVGFusion)
    nn.Module.__init__(model)
    model.args = runtime.recipe(tmp_path, variant)
    model.visual_exchange_mode = model.args.visual_exchange
    model.clip, model.other = nn.Linear(4, 4), nn.Linear(4, 4)
    model.cross_fusion_layers_vt = nn.ModuleList([nn.Linear(4, 4)])
    model.cross_fusion_layers_tv = nn.ModuleList([nn.Linear(4, 4)])
    if variant != 'c1_continue':
        model.region_anchor_exchange = nn.Linear(4, 4)
    model.infmae_local_center = None
    model.clip.set_adapter = lambda _: [p.requires_grad_(True) for p in model.clip.parameters()]
    model._configure_exchange_tuning()
    model.train()
    model._select_grounding_clip_adapter('lora_ir')
    runtime.assert_scope(model, model.args)
    assert not model.training and not model.other.training and not model.clip.training
    if variant != 'c1_continue':
        assert model.region_anchor_exchange.training
    optimizer, _ = runtime.optimizer_for(model, model.args)
    assert optimizer.param_groups[0]['initial_lr'] == 1e-5
    record = runtime.delta_record(model, model.args, 0, {'accu': .5, 'accu_07': .3, 'miou': .4}, .5, {})
    with torch.no_grad():
        for name, p in model.named_parameters():
            if runtime.selected_name(name, model.args):
                p.add_(1.)
    runtime.restore_delta(model, model.args, record)
    assert all(torch.equal(t, model.state_dict()[n]) for n, t in record['model_delta'].items())
    bad = copy.deepcopy(record)
    bad['model_delta']['other.weight'] = model.other.weight.detach().clone()
    with pytest.raises(ValueError):
        runtime.restore_delta(model, model.args, bad)


def test_anchor_loss_uses_the_real_box_objective_without_reliability_auxiliary(tmp_path):
    from utils.loss_utils import trans_vg_loss
    args = runtime.recipe(tmp_path)
    target = torch.tensor([[.4, .5, .2, .3]])
    prediction = torch.tensor([[.45, .55, .22, .28]], requires_grad=True)
    losses = trans_vg_loss(args, prediction, target, None, None)
    assert set(losses) == {'loss_bbox', 'loss_giou'}
    sum(losses.values()).backward()
    assert torch.isfinite(prediction.grad).all()


def test_new_structure_must_beat_a_matched_continuation_and_protect_quality():
    def metrics(hits=476, epoch=1):
        return dict(accu=hits / 608, accu_07=.58, miou=.66, miou_weak=.63, epoch=epoch)
    runs = {v: metrics(481) for v in runtime.VARIANTS}
    assert runtime.choose_promotion(metrics(), runs)['selected_variant'] is None
    runs['anchor_text_separate'] = metrics(483)
    assert runtime.choose_promotion(metrics(), runs)['selected_variant'] == 'anchor_text_separate'
    runs['anchor_text_separate']['miou_weak'] -= .01
    assert runtime.choose_promotion(metrics(), runs)['selected_variant'] is None


def test_trained_best_checkpoint_recovers_after_interrupted_commit(tmp_path):
    model = nn.Linear(2, 2)
    args = SimpleNamespace(experiment_case='probe', exchange_base_checkpoint='probe', exchange_base_sha256='a')
    # Selection journals do not depend on the model's architecture.
    record = dict(epoch=0, weight=model.weight.detach().clone())
    runtime.save_selection(tmp_path, 'trained_best', record, {'accu': .5})
    runtime.save_selection(tmp_path, 'trained_best', dict(epoch=1), {'accu': .6})
    runtime.recover_selection(tmp_path, 0)
    assert torch.load(tmp_path / 'trained_best_checkpoint.pth', weights_only=False)['epoch'] == 0


def test_selection_recovers_if_only_the_old_weight_was_moved(tmp_path):
    runtime.save_selection(tmp_path, 'trained_best', dict(epoch=0), {'accu': .5})
    (tmp_path / 'trained_best_checkpoint.pth').replace(tmp_path / 'previous_trained_best_checkpoint.pth')
    runtime.recover_selection(tmp_path, 0)
    assert torch.load(tmp_path / 'trained_best_checkpoint.pth', weights_only=False)['epoch'] == 0


def test_gpu_absence_records_waiting_without_claiming_training(tmp_path, monkeypatch):
    from script_train.clip_anchor_control import control
    runtime.write_json(tmp_path / 'status.json', {'phase': 'prepared', 'jobs': {}, 'completed_epochs': 0})
    monkeypatch.setattr(control, 'verify_snapshot', lambda _: {})
    def missing(_):
        raise RuntimeError('NVIDIA driver unavailable')
    monkeypatch.setattr(control, 'gpu_pair', missing)
    assert not control.launch(tmp_path)
    status = json.loads((tmp_path / 'status.json').read_text())
    assert status['phase'] == 'waiting_for_gpu'
    assert not status['training_started'] and not status['controller_active']


def test_gpu_selection_respects_other_users_and_requires_two_distinct_cards(monkeypatch):
    from script_train.clip_anchor_control import control
    monkeypatch.setattr(control.subprocess, 'run', lambda *a, **kw: SimpleNamespace(
        returncode=0, stdout='0, 9000, 30000\n1, 10, 23000\n2, 100, 22500\n', stderr=''))
    assert control.gpu_pair() == '1,2'
    with pytest.raises(RuntimeError):
        control.gpu_pair('0,1')
    with pytest.raises(RuntimeError):
        control.gpu_pair('1,1')
