from types import SimpleNamespace
import json

import pytest
import torch

from script_train.clip_scale_geometry_control import runtime as r
from script_train.clip_scale_geometry_control import data
from utils.relative_geometry_losses import relative_geometry_losses


def args():
    return SimpleNamespace(imsize=224, relative_geometry_floor_pixels=4.,
                           relative_center_weight=.25, relative_shape_weight=.05)


def test_exact_boxes_have_zero_auxiliary_loss_and_zero_gradient():
    target = torch.tensor([[.5, .5, .02, .03]])
    prediction = target.clone().requires_grad_(True)
    values = relative_geometry_losses(args(), prediction, target)
    sum(values.values()).backward()
    assert all(float(value) == 0. for value in values.values())
    assert torch.equal(prediction.grad, torch.zeros_like(prediction))


def test_equal_relative_offsets_have_equal_loss_above_the_floor():
    target = torch.tensor([[.5, .5, .1, .1], [.5, .5, .2, .2]])
    prediction = target.clone()
    prediction[:, 0] += torch.tensor([.02, .04])
    first = relative_geometry_losses(args(), prediction[:1], target[:1])
    second = relative_geometry_losses(args(), prediction[1:], target[1:])
    assert torch.allclose(first['loss_relative_center'], second['loss_relative_center'])


def test_tiny_targets_produce_finite_gradients_toward_correct_center_and_size():
    target = torch.tensor([[.5, .5, 1e-7, 1e-7]])
    prediction = torch.tensor([[.6, .4, .02, .02]], requires_grad=True)
    sum(relative_geometry_losses(args(), prediction, target).values()).backward()
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad[0, 0] > 0 and prediction.grad[0, 1] < 0
    assert (prediction.grad[0, 2:] > 0).all()


def test_scale_protection_is_disabled_for_validation_and_test(monkeypatch):
    def initialize(self, split, parameters):
        self.sizes = [56, 224]
    monkeypatch.setattr(data.NativeDataset, '__init__', initialize)
    a = SimpleNamespace(target_scale_floor_pixels=8., aug_translate=True)
    for split, expected in [('train', 8.), ('val', 0.), ('test', 0.)]:
        dataset = data.ScaleGeometryDataset(split, a)
        assert dataset.finish.transforms[0].minimum_target_side == expected


def test_factorial_controls_change_only_scale_and_geometry_flags(tmp_path):
    recipes = [r.recipe(tmp_path, name) for name in r.VARIANTS]
    assert [(a.target_scale_floor_pixels, a.relative_geometry_enabled) for a in recipes] == [
        (0., False), (8., False), (0., True), (8., True)]
    assert all(a.batch_size * a.grad_accum_steps * a.world_size == 96 for a in recipes)
    assert all(a.epochs == 30 and a.projection_lr == 2e-6 and not a.reobserve_enabled for a in recipes)
    assert all(r.selected_name('base.bbox_embed.layers.0.weight', a) for a in recipes)
    assert not any(r.selected_name('base.clip.vision_model.weight', a) for a in recipes)


def test_promotion_does_not_attribute_plain_continuation_to_new_method():
    baseline = dict(epoch=-1, accu=476/608, accu_07=.58, miou=.66,
                    miou_weak=.63, accu_normal=.97, accu_small=.64)
    runs = {name: dict(baseline, epoch=3, accu=478/608) for name in r.VARIANTS}
    assert r.choose_promotion(baseline, runs)['selected_variant'] is None


def test_relative_losses_reach_the_actual_training_loss_dispatch():
    from utils.loss_utils import trans_vg_loss
    a = SimpleNamespace(relative_geometry_enabled=True, **vars(args()),
                        use_mask_loss=False, use_contrastive_loss=False, use_rtcc_constrain_loss=False)
    prediction = torch.tensor([[.5, .5, .1, .1]], requires_grad=True)
    target = torch.tensor([[.55, .5, .1, .1]])
    result = trans_vg_loss(a, prediction, target, None, None)
    assert result['loss_relative_center'] > 0
    sum(v for k, v in result.items() if k.startswith('loss_')).backward()
    assert torch.isfinite(prediction.grad).all()


def write_completed_runs(root, count):
    for variant in r.VARIANTS:
        directory = root / 'explore' / f'{variant}_seed13'
        directory.mkdir(parents=True)
        (directory / 'initial_validation.json').write_text(json.dumps({'accu': .6}))
        (directory / 'best.json').write_text(json.dumps({'epoch': 4, 'accu': .8}))
        rows = [{'epoch': epoch, 'val_accu': .8 if epoch == 4 else .7} for epoch in range(count)]
        (directory / 'log.txt').write_text(''.join(json.dumps(row) + '\n' for row in rows))


def test_report_uses_registered_thirty_epochs_instead_of_inherited_twenty(tmp_path):
    write_completed_runs(tmp_path, 30)
    result = r.collect_best_runs(tmp_path)
    assert set(result) == set(r.VARIANTS)
    assert all(value['epoch'] == 4 for value in result.values())


def test_report_still_rejects_a_run_missing_its_last_registered_epoch(tmp_path):
    write_completed_runs(tmp_path, 29)
    with pytest.raises(ValueError, match='Incomplete exploration'):
        r.collect_best_runs(tmp_path)
