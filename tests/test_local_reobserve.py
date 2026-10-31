from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).parents[1]))
from models.local_reobserve import native_crops, sample_native, ReobserveHead, LocalReobserveGrounder
from script_train.clip_reobserve_control.data import valid_rectangle, NativeBatch
from script_train.clip_reobserve_control import runtime
from utils.reobserve_losses import reobserve_losses

torch.set_num_threads(2)


def test_round_trip_coordinates_respect_translation_and_non_square_native_size():
    mask = torch.ones(224, 224, dtype=torch.bool)
    mask[22:201, :] = False
    rectangle = valid_rectangle(mask, (512, 640))
    assert torch.allclose(rectangle, torch.tensor([0., 22 / 224, 1., .8]))
    original = torch.tensor([[.7, .4, .05, .03]])
    coarse = torch.cat((original[:, :2] * rectangle[None, 2:] + rectangle[None, :2],
                        original[:, 2:] * rectangle[None, 2:]), -1)
    native, canvas = native_crops(coarse, rectangle[None], floor=0.)
    assert torch.allclose(native[:, 0, :2], original[:, :2])
    assert torch.allclose(canvas[:, 0, :2], coarse[:, :2])
    assert torch.allclose(native[:, 0, 2:], original[:, 2:] * 3)


def test_sampling_native_pixels_does_not_read_batch_padding_and_has_geometry_gradients():
    image = torch.arange(4 * 8 * 12, dtype=torch.int32).reshape(1, 4, 8, 12).remainder(255).to(torch.uint8)
    packed = torch.zeros(1, 4, 12, 16, dtype=torch.uint8)
    packed[:, :, :8, :12] = image
    crop = torch.tensor([[.5, .5, 1., 1.]], requires_grad=True)
    out = sample_native(packed, torch.tensor([[8., 12.]]), crop, 0, 12)
    expected = F.interpolate(image[:, :3].float() / 255, (12, 12), mode='bilinear', align_corners=False)
    expected = (expected - torch.tensor([.631, .6401, .632])[None, :, None, None]) / torch.tensor([.2152, .227, .2439])[None, :, None, None]
    assert torch.allclose(out, expected, atol=3e-6)
    out.square().mean().backward()
    assert torch.isfinite(crop.grad).all() and crop.grad.abs().sum() > 0
    changed = packed.clone()
    changed[:, :, 8:] = 255
    changed[:, :, :, 12:] = 255
    assert torch.equal(out.detach(), sample_native(changed, torch.tensor([[8., 12.]]), crop.detach(), 0, 12))


def fixture():
    torch.manual_seed(13)
    head = ReobserveHead(hidden=32)
    coarse = torch.tensor([[.5, .5, .1, .08], [.6, .4, .2, .15]])
    visual = [torch.randn(2, 197, 512) for _ in range(2)]
    text = torch.randn(2, 8, 512)
    padding = torch.zeros(2, 8, dtype=torch.bool)
    query = head.queries(torch.randn(2, 512), coarse, visual, text, padding)
    return head, coarse, visual, text, padding, query


def test_zero_local_output_is_exactly_the_global_prediction():
    head, coarse, local, _, _, query = fixture()
    for adaptive in (False, True):
        policy = head.policy(query)
        native, canvas = native_crops(coarse, torch.tensor([[0., 0., 1., 1.]]).expand(2, -1), policy if adaptive else policy * 0.)
        result, auxiliary = head.decode(query, coarse, canvas, local)
        assert torch.equal(result, coarse)
        assert torch.equal(auxiliary['reobserve_proposals'], coarse[:, None].expand(-1, 2, -1))
        assert (native[..., :2] - native[..., 2:] / 2 >= 0).all()
        assert (native[..., :2] + native[..., 2:] / 2 <= 1).all()


def test_text_condition_changes_learned_read_geometry():
    head, coarse, visual, text, padding, _ = fixture()
    nn.init.normal_(head.policy[-1].weight, std=.1)
    reg = torch.randn(2, 512)
    a = head.policy(head.queries(reg, coarse, visual, text, padding))
    b = head.policy(head.queries(reg, coarse, visual, -text, padding))
    assert not torch.equal(a, b)
    loss = a.square().mean()
    loss.backward()
    assert head.text_proj.weight.grad.abs().sum() > 0


@pytest.mark.parametrize('mode', ['fixed_local', 'text_adaptive', 'text_consistent'])
def test_no_crop_contains_target_does_not_create_truncated_gt_or_nan(mode):
    target = torch.tensor([[.8, .8, .1, .1]])
    proposals = torch.tensor([[[.2, .2, .1, .1], [.2, .2, .1, .1]]], requires_grad=True)
    aux = {'reobserve_crops': torch.tensor([[[.2, .2, .2, .2], [.2, .2, .2, .2]]]),
           'reobserve_proposals': proposals, 'reobserve_accept_logits': torch.zeros(1, requires_grad=True)}
    losses = reobserve_losses(SimpleNamespace(reobserve_mode=mode), aux, target)
    assert losses['loss_reobserve_local'] == 0
    assert losses['reobserve_coverage_fraction'] == 0
    sum(v for k, v in losses.items() if k.startswith('loss_')).backward()
    assert torch.equal(proposals.grad, torch.zeros_like(proposals))
    assert torch.isfinite(aux['reobserve_accept_logits'].grad).all()


def test_consistency_compares_mapped_canvas_coordinates():
    target = torch.tensor([[.5, .5, .1, .1]])
    proposals = torch.tensor([[[.48, .5, .1, .1], [.52, .5, .1, .1]]], requires_grad=True)
    aux = {'reobserve_crops': torch.tensor([[[.5, .5, .5, .5], [.5, .5, .6, .6]]]),
           'reobserve_proposals': proposals, 'reobserve_accept_logits': torch.zeros(1, requires_grad=True)}
    losses = reobserve_losses(SimpleNamespace(reobserve_mode='text_consistent'), aux, target)
    assert losses['loss_reobserve_consistency'] > 0
    losses['loss_reobserve_consistency'].backward()
    assert proposals.grad[:, 0, 0] < 0 and proposals.grad[:, 1, 0] > 0


def test_gt_is_rejected_by_model_forward():
    model = object.__new__(LocalReobserveGrounder)
    nn.Module.__init__(model)
    with pytest.raises(ValueError, match='GT boxes'):
        model(None, None, target_boxes=torch.ones(1, 4))


def test_registered_controls_have_same_base_scope_and_effective_batch(tmp_path):
    recipes = [runtime.recipe(tmp_path, v) for v in runtime.VARIANTS]
    assert all(a.batch_size * a.grad_accum_steps * a.world_size == 96 for a in recipes)
    assert all(not a.exchange_train_new_only and not a.exchange_projection_tuning for a in recipes)
    assert all(a.exchange_base_sha256 == recipes[0].exchange_base_sha256 for a in recipes)
    assert all(runtime.selected_name('base.bbox_embed.layers.0.weight', a) for a in recipes)
    assert all(not runtime.selected_name('base.clip.weight', a) for a in recipes)


def test_delta_restore_rejects_changes_to_the_frozen_backbone(tmp_path):
    args = runtime.recipe(tmp_path, 'fixed_local')
    model = nn.Module()
    model.base = nn.Module()
    model.base.bbox_embed = nn.Linear(3, 2)
    model.base.clip = nn.Linear(3, 2)
    model.refiner = nn.Linear(3, 2)
    for n, p in model.named_parameters():
        p.requires_grad_(runtime.selected_name(n, args))
    state = runtime.delta_record(model, args, 0, {'accu': .5, 'accu_07': .3, 'miou': .4}, .5, {})
    with torch.no_grad():
        model.refiner.weight.add_(1.)
    runtime.restore_delta(model, args, state)
    assert torch.equal(model.refiner.weight, state['model_delta']['refiner.weight'])
    state['model_delta']['base.clip.weight'] = model.base.clip.weight.detach().clone()
    with pytest.raises(ValueError, match='outside'):
        runtime.restore_delta(model, args, state)


def test_adaptive_reading_must_outperform_fixed_local_reading_to_pass():
    def metrics(hits, epoch=0):
        return dict(epoch=epoch, accu=hits / 608, accu_07=.6, miou=.67, miou_weak=.64,
                    accu_normal=.98, accu_small=.66)
    baseline = metrics(476)
    baseline['accu_small'] = .64
    runs = {v: metrics(480) for v in runtime.VARIANTS}
    result = runtime.choose_promotion(baseline, runs)
    assert not result['checks']['text_adaptive']['passed']
    runs['c1_continue'] = metrics(477)
    runs['text_consistent'] = metrics(482)
    result = runtime.choose_promotion(baseline, runs)
    assert result['selected_variant'] == 'text_consistent'
