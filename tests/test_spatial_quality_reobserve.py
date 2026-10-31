from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parents[1]))
from models.local_reobserve import native_crops
from models.spatial_quality_reobserve import SpatialQualityHead, SpatialQualityGrounder
from utils.spatial_quality_losses import spatial_quality_losses
from script_train.clip_spatial_quality_control import runtime

torch.set_num_threads(2)


def fixture(mode):
    torch.manual_seed(17)
    model = SpatialQualityHead(hidden=32, mode=mode)
    coarse = torch.tensor([[.5, .5, .12, .1], [.65, .4, .08, .06]])
    local = [torch.randn(2, 197, 512) for _ in range(2)]
    query = torch.randn(2, 2, 32)
    _, crops = native_crops(coarse, torch.tensor([[0., 0., 1., 1.]]).expand(2, -1))
    return model, coarse, query, crops, local


@pytest.mark.parametrize('mode', ['spatial_uniform', 'fixed_benefit', 'adaptive_benefit'])
def test_spatial_initialization_is_exactly_coarse_with_normalized_distributions(mode):
    model, coarse, query, crops, local = fixture(mode)
    final, aux = model.decode(query, coarse, crops, local)
    assert torch.equal(final, coarse)
    assert torch.allclose(aux['spatial_posterior'].sum(-1), torch.ones(2, 2))
    assert torch.equal(aux['reobserve_proposals'][..., 2:], coarse[:, None, 2:].expand(-1, 2, -1))
    assert torch.equal(aux['reobserve_proposals'][..., :2], coarse[:, None, :2].expand(-1, 2, -1))


def test_negative_utility_rejects_local_evidence_even_when_the_crop_covers_target():
    model, coarse, query, crops, local = fixture('fixed_benefit')
    nn.init.normal_(model.score.weight, std=.2)
    with torch.no_grad():
        model.quality[-1].bias.fill_(-.1)
    final, aux = model.decode(query, coarse, crops, local)
    assert not torch.equal(aux['reobserve_proposals'][..., :2], coarse[:, None, :2].expand(-1, 2, -1))
    assert torch.equal(final, coarse)
    assert not aux['spatial_weights'].any()


def test_positive_utility_accepts_center_evidence_while_preserving_width_and_height():
    model, coarse, query, crops, local = fixture('fixed_benefit')
    nn.init.normal_(model.score.weight, std=.2)
    with torch.no_grad():
        model.quality[-1].bias.fill_(.1)
    final, aux = model.decode(query, coarse, crops, local)
    assert not torch.equal(final[:, :2], coarse[:, :2])
    assert torch.equal(final[:, 2:], coarse[:, 2:])
    assert (aux['spatial_weights'].sum(-1) < 1).all()


def test_spatial_evidence_changes_when_local_visual_tokens_change():
    model, coarse, query, crops, local = fixture('spatial_uniform')
    nn.init.normal_(model.score.weight, std=.2)
    a, aa = model.decode(query, coarse, crops, local)
    b, bb = model.decode(query, coarse, crops, [-x for x in local])
    assert not torch.equal(aa['spatial_posterior'], bb['spatial_posterior'])
    assert not torch.equal(a[:, :2], b[:, :2])


def test_localization_supervision_updates_spatial_scores_without_regressing_size():
    model, coarse, query, crops, local = fixture('spatial_uniform')
    target = coarse.clone()
    target[:, 0] += .03
    _, aux = model.decode(query, coarse, crops, local)
    losses = spatial_quality_losses(SimpleNamespace(reobserve_mode='spatial_uniform'), aux, target)
    sum(v for k, v in losses.items() if k.startswith('loss_')).backward()
    assert torch.isfinite(model.score.weight.grad).all() and model.score.weight.grad.abs().sum() > 0
    assert 'loss_spatial_utility' not in losses


def test_spatial_label_loss_cannot_move_crop_geometry_without_image_evidence():
    model, coarse, query, crops, local = fixture('spatial_uniform')
    nn.init.normal_(model.score.weight, std=.1)
    crops = crops.detach().requires_grad_(True)
    local = [x.detach().requires_grad_(True) for x in local]
    target = coarse.clone()
    target[:, 0] += .03
    _, aux = model.decode(query, coarse, crops, local)
    losses = spatial_quality_losses(SimpleNamespace(reobserve_mode='spatial_uniform'), aux, target)
    losses['loss_spatial_center'].backward()
    assert crops.grad is None
    assert all(x.grad is not None and x.grad.abs().sum() > 0 for x in local)


def test_utility_target_rewards_actual_iou_gain_not_coverage():
    model, coarse, query, crops, local = fixture('fixed_benefit')
    target = coarse.clone()
    target[:, 0] += .03
    _, aux = model.decode(query, coarse, crops, local)
    proposals = coarse[:, None].expand(-1, 2, -1).clone()
    proposals[:, 0] = target
    proposals[:, 1, 0] -= .1
    predicted = torch.zeros(2, 2, requires_grad=True)
    aux['reobserve_proposals'], aux['spatial_predicted_gain'] = proposals, predicted
    losses = spatial_quality_losses(SimpleNamespace(reobserve_mode='fixed_benefit'), aux, target)
    losses['loss_spatial_utility'].backward()
    assert (predicted.grad[:, 0] < 0).all() and (predicted.grad[:, 1] > 0).all()


def test_group_clipping_does_not_shrink_reader_updates_due_to_backbone_gradients():
    from engine import _clip_training_gradients
    a, b = nn.Parameter(torch.zeros(1)), nn.Parameter(torch.zeros(1))
    model = nn.ParameterList([a, b])
    optimizer = torch.optim.AdamW([{'params': [a]}, {'params': [b]}])
    a.grad, b.grad = torch.tensor([1000.]), torch.tensor([1.])
    _clip_training_gradients(SimpleNamespace(clip_optimizer_groups_separately=True), model, optimizer, 1.)
    assert .99 < a.grad.item() <= 1. and .99 < b.grad.item() <= 1.


def test_gt_is_rejected_by_the_shared_forward_path():
    model = object.__new__(SpatialQualityGrounder)
    nn.Module.__init__(model)
    with pytest.raises(ValueError, match='GT boxes'):
        model(None, None, target_boxes=torch.ones(1, 4))


def test_all_controls_share_the_base_recipe_and_preserve_effective_batch(tmp_path):
    recipes = [runtime.recipe(tmp_path, v) for v in runtime.VARIANTS]
    assert all(a.batch_size * a.grad_accum_steps * a.world_size == 96 for a in recipes)
    assert all(a.clip_optimizer_groups_separately and a.projection_lr == 2e-6 for a in recipes)
    assert all(runtime.selected_name('base.bbox_embed.layers.0.weight', a) for a in recipes)
    assert not runtime.selected_name('refiner.quality.0.weight', recipes[1])
    assert all(runtime.selected_name('refiner.quality.0.weight', a) for a in recipes[2:])
