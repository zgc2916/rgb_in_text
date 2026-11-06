"""Check positional layout, true native resizing, and matched selection rules."""
import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from models.dense_grid import grounding_positions, DenseTextReliability, DenseGridGrounder, parameter_group
from script_train.clip_dense_grid_control import runtime as r, data


def test_224_positions_are_exact_and_unused_tail_is_never_reinterpreted():
    weight = torch.randn(470, 8)
    assert torch.equal(grounding_positions(weight, 14), weight[:275])
    dense = grounding_positions(weight, 21)
    assert dense.shape == (520, 8)
    assert torch.equal(dense[:2], weight[:2])
    assert torch.equal(dense[-77:], weight[198:275])
    weight[275:] += 1000
    assert torch.equal(dense, grounding_positions(weight, 21))


def test_position_interpolation_preserves_spatial_constants_and_gradients():
    weight = torch.ones(470, 4, requires_grad=True)
    output = grounding_positions(weight, 21)
    assert torch.allclose(output, torch.ones_like(output))
    output.sum().backward()
    assert (weight.grad[2:198] > 0).all()
    assert torch.equal(weight.grad[275:], torch.zeros_like(weight.grad[275:]))
    with pytest.raises(ValueError):
        grounding_positions(weight[:200], 21)


def fixture():
    torch.manual_seed(71)
    return (torch.randn(10, 2, 8), torch.randn(10, 2, 8), torch.randn(10, 2, 8),
            torch.randn(2, 8), torch.tensor([[False]*8+[True], [False]*9]))


def test_zero_gate_preserves_dense_model_and_real_output_gradients_exist():
    module = DenseTextReliability()
    inputs = fixture()
    result = module(*inputs)
    assert torch.equal(result, inputs[0])
    result.square().sum().backward()
    assert module.predict[-1].weight.grad.abs().sum() > 0
    assert sum(p.numel() for p in module.parameters()) == 257


def test_gate_never_modifies_cls_or_padding_and_padding_cannot_change_valid_cells():
    module = DenseTextReliability()
    nn.init.normal_(module.predict[-1].weight, std=.3)
    inputs = fixture()
    result = module(*inputs)
    assert torch.equal(result[0], inputs[0][0])
    assert torch.equal(result[-1, 0], inputs[0][-1, 0])
    changed = [x.clone() for x in inputs]
    changed[1][-1, 0] += 100
    changed[2][-1, 0] -= 100
    assert torch.equal(result[:-1, 0], module(*changed)[:-1, 0])
    assert (result[1:] - inputs[0][1:]).abs().max() > 0
    with pytest.raises(ValueError):
        module(*inputs[:3], None, inputs[-1])


def test_new_parameters_do_not_open_language_or_early_visual_layers():
    assert parameter_group('base.dense_reliability.predict.2.weight') == 'reliability'
    assert parameter_group('base.vl_pos_embed.weight') is None
    assert parameter_group('base.clip.base_model.model.text_model.encoder.layers.8.self_attn.q_proj.lora_A.lora_rgb.weight') is None


def test_gt_is_rejected_by_dense_forward():
    model = object.__new__(DenseGridGrounder)
    nn.Module.__init__(model)
    with pytest.raises(ValueError, match='GT boxes'):
        model(None, None, target_boxes=torch.ones(1, 4))


def test_resolution_data_finishes_from_native_pixels_and_evaluation_has_no_augmentation(monkeypatch):
    def initialize(self, split, args):
        self.source = object()
    monkeypatch.setattr(data.NativeDataset, '__init__', initialize)
    a = SimpleNamespace(imsize=336, aug_scale=True, aug_translate=True)
    train, val = data.ResolutionDataset('train', a), data.ResolutionDataset('val', a)
    assert train.sizes == [336,294,252,210,168,126,84]
    assert val.sizes == [336]
    assert train.finish.transforms[-1].size == 336
    assert not val.finish.transforms[-1].aug_translate


def test_variants_share_batch_optimizer_scope_and_original_checkpoint(tmp_path):
    recipes = [r.recipe(tmp_path, v) for v in r.VARIANTS]
    assert [a.imsize for a in recipes] == [224,336,336]
    assert [a.dense_text_gate for a in recipes] == [False,False,True]
    assert len({a.exchange_base_sha256 for a in recipes}) == 1
    assert all(a.epochs == 20 and a.batch_size*a.grad_accum_steps*a.world_size == 96 for a in recipes)
    assert all(not a.relative_geometry_enabled and not a.reobserve_enabled for a in recipes)


def test_resolution_gain_cannot_be_attributed_to_text_gate():
    base = dict(r.reference(13)['validation'], epoch=1)
    runs = {v+'_seed13':copy.deepcopy(base) for v in r.VARIANTS}
    for v in ('dense_336','dense_text_336'):
        runs[v+'_seed13']['accu'] += 6/608
        runs[v+'_seed13']['accu_small'] += .02
    result = r.choose_promotion(runs)
    assert result['selected_variant'] == 'dense_336'
    assert not result['text_reliability_passed']
    runs['dense_text_336_seed13']['accu'] += 2/608
    runs['dense_text_336_seed13']['accu_small'] += .01
    assert r.choose_promotion(runs)['selected_variant'] == 'dense_text_336'


def test_one_good_seed_cannot_pass_replication():
    runs={f'{v}_seed{s}':copy.deepcopy(r.reference(s)['validation']) for s in r.SEEDS for v in r.VARIANTS}
    runs['dense_336_seed13']['accu'] += .03
    runs['dense_336_seed13']['accu_small'] += .03
    assert not r.replication_gate(runs,'dense_336')['passed']


def test_dense_delta_restores_gate_and_rejects_frozen_parameter_or_wrong_format(tmp_path):
    a = r.recipe(tmp_path, 'dense_text_336')
    model = nn.Module()
    model.base = nn.Module()
    model.base.dense_reliability = DenseTextReliability()
    model.base.frozen = nn.Linear(2, 2)
    record = r.delta_record(model, a, 0, {'accu':.8,'accu_07':.6,'miou':.67}, .8, {})
    before = {n: t.clone() for n,t in record['model_delta'].items()}
    with torch.no_grad():
        model.base.dense_reliability.predict[-1].weight.add_(1)
    r.restore_delta(model, a, record)
    assert all(torch.equal(t, model.state_dict()[n]) for n,t in before.items())
    bad = copy.deepcopy(record)
    bad['model_delta']['base.frozen.weight'] = model.base.frozen.weight.detach().clone()
    with pytest.raises(ValueError):
        r.restore_delta(model, a, bad)
    bad = copy.deepcopy(record)
    bad['format'] = 'another_resolution_experiment'
    with pytest.raises(ValueError):
        r.restore_delta(model, a, bad)


def test_optimizers_cover_gate_separately_without_opening_frozen_parameters(tmp_path):
    a = r.recipe(tmp_path, 'dense_text_336')
    model = nn.Module()
    model.base = nn.Module()
    model.base.dense_reliability = DenseTextReliability()
    model.base.frozen = nn.Linear(2, 2)
    for name,p in model.named_parameters():
        p.requires_grad_(r.selected_name(name,a))
    optimizer, _ = r.optimizer_for(model,a)
    assert [g['name'] for g in optimizer.param_groups] == ['reliability']
    assert optimizer.param_groups[0]['initial_lr'] == 1e-4
    assert {id(p) for g in optimizer.param_groups for p in g['params']} == {
        id(p) for p in model.base.dense_reliability.parameters()}
