import hashlib
import inspect

import pytest
import torch

from models.center_evidence import (
    CenterEvidenceHead, center_heatmap_loss, masked_spatial_softmax, normalized_grid,
)

torch.set_num_threads(2)


def inputs(size=32):
    torch.manual_seed(41)
    images = torch.randn(2, 4, size, size)
    padding = torch.zeros(2, size, size, dtype=torch.bool)
    padding[0, :, -4:] = True
    text, reg = torch.randn(2, 512), torch.randn(2, 512)
    boxes = torch.tensor([[.32, .60, .20, .12], [.70, .40, .12, .15]])
    return images, padding, text, reg, boxes


def model(mode='fixed'):
    return CenterEvidenceHead(mode, image_size=32, hidden=16,
                              min_window_pixels=16, min_shift_pixels=4, max_shift_pixels=8)


@pytest.mark.parametrize('mode', ['fixed', 'text_gate'])
def test_zero_init_preserves_boxes_and_no_ground_truth_enters_forward(mode):
    head = model(mode)
    args = inputs()
    output, aux = head(*args)
    assert torch.equal(output, args[-1])
    assert torch.equal(output[:, 2:], args[-1][:, 2:])
    assert torch.allclose(aux['probability'].sum((1, 2)), torch.ones(2))
    assert torch.allclose(aux['modality_weights'], torch.full((2, 2), .5))
    assert 'target_boxes' not in inspect.signature(head.forward).parameters
    with pytest.raises(TypeError):
        head(*args, target_boxes=args[-1])


def test_text_reg_and_both_images_affect_spatial_evidence_independently():
    head = model('text_gate').eval()
    images, padding, text, reg, boxes = inputs()
    _, initial = head(images, padding, text, reg, boxes)
    changed_text = text.clone(); changed_text[0] *= -1
    _, text_result = head(images, padding, changed_text, reg, boxes)
    changed_reg = reg.clone(); changed_reg[0] *= -1
    _, reg_result = head(images, padding, text, changed_reg, boxes)
    assert not torch.allclose(initial['heatmap_logits'][0], text_result['heatmap_logits'][0])
    assert not torch.allclose(initial['heatmap_logits'][0], reg_result['heatmap_logits'][0])
    assert torch.equal(initial['heatmap_logits'][1], text_result['heatmap_logits'][1])
    for channel_slice in (slice(0, 3), slice(3, 4)):
        changed_images = images.clone()
        changed_images[0, channel_slice, 4:24, 4:24] += 5 * torch.randn_like(changed_images[0, channel_slice, 4:24, 4:24])
        _, altered = head(changed_images, padding, text, reg, boxes)
        assert not torch.allclose(initial['heatmap_logits'][0], altered['heatmap_logits'][0])
        assert torch.equal(initial['heatmap_logits'][1], altered['heatmap_logits'][1])


def test_fixed_weights_and_learned_text_reg_gate_are_distinct():
    fixed, dynamic = model('fixed'), model('text_gate')
    dynamic.load_state_dict(fixed.state_dict())
    assert not any(parameter.requires_grad for parameter in fixed.route.parameters())
    assert all(parameter.requires_grad for parameter in dynamic.route.parameters())
    with torch.no_grad():
        dynamic.route[-1].weight[0, 0] = 1.
        dynamic.route[-1].weight[1, 1] = -1.
    images, padding, text, reg, boxes = inputs()
    _, uniform = fixed(images, padding, text, reg, boxes)
    _, routed = dynamic(images, padding, text, reg, boxes)
    _, changed = dynamic(images, padding, -text, -reg, boxes)
    assert torch.equal(uniform['modality_weights'], torch.full((2, 2), .5))
    assert torch.allclose(routed['modality_weights'].sum(-1), torch.ones(2))
    assert not torch.allclose(routed['modality_weights'], changed['modality_weights'])


def test_padding_pixels_cannot_change_predictions_or_heatmap_and_empty_is_safe():
    head = model('text_gate').eval()
    images, padding, text, reg, boxes = inputs()
    padding[1] = True
    with torch.no_grad():head.offset[-1].bias.fill_(.7)
    output, aux = head(images, padding, text, reg, boxes)
    changed = images.clone()
    changed[padding[:, None].expand_as(images)] += 1000
    output2, aux2 = head(changed, padding, text, reg, boxes)
    assert torch.equal(output, output2)
    assert torch.equal(aux['heatmap_logits'], aux2['heatmap_logits'])
    assert not aux['has_evidence'][1]
    assert torch.equal(output[1], boxes[1])
    assert aux['probability'][1].count_nonzero() == 0
    assert torch.isfinite(output).all() and torch.isfinite(aux['probability']).all()


def test_pixel_center_coordinates_and_masked_distribution():
    grid = normalized_grid(8, 8, 'cpu', torch.float32)
    assert torch.equal(grid[0, 0], torch.tensor([.0625, .0625]))
    assert torch.equal(grid[7, 7], torch.tensor([.9375, .9375]))
    scores = torch.full((2, 8, 8), -100.)
    scores[0, 3, 5] = 100
    valid = torch.ones_like(scores, dtype=torch.bool); valid[1] = False
    distribution, has_evidence = masked_spatial_softmax(scores, valid)
    center = (distribution[..., None] * grid).sum((1, 2))
    assert torch.allclose(center[0], grid[3, 5], atol=1e-6)
    assert torch.equal(center[1], torch.zeros(2))
    assert torch.equal(has_evidence, torch.tensor([True, False]))


def test_bounded_center_only_correction_and_real_gradients():
    head = model('text_gate')
    images, padding, text, reg, boxes = inputs()
    images.requires_grad_(); text.requires_grad_(); reg.requires_grad_(); boxes.requires_grad_()
    with torch.no_grad():
        head.offset[-1].weight.normal_(std=.2)
        head.offset[-1].bias.fill_(.2)
        head.route[-1].weight.normal_(std=.1)
    output, aux = head(images, padding, text, reg, boxes)
    assert torch.equal(output[:, 2:], boxes[:, 2:])
    assert (aux['center_delta'].abs() <= 8/32 + 1e-6).all()
    assert ((output[:, :2] >= 0) & (output[:, :2] <= 1)).all()
    target = boxes.detach().clone(); target[:, 0] += .03
    loss = (output-target).square().sum() + center_heatmap_loss(aux, target)
    loss.backward()
    for tensor in (images.grad, text.grad, reg.grad, boxes.grad,
                   head.rgb_encoder.stem[0][0].weight.grad,
                   head.tir_encoder.stem[0][0].weight.grad, head.route[-1].weight.grad):
        assert tensor is not None and torch.isfinite(tensor).all() and tensor.abs().sum() > 0
    assert images.grad[padding[:, None].expand_as(images)].count_nonzero() == 0


def test_minimum_shift_is_configurable_for_tiny_boxes():
    default = CenterEvidenceHead('fixed')
    assert default.min_shift == pytest.approx(12/224)
    assert default.max_shift == pytest.approx(32/224)
    head = CenterEvidenceHead('fixed', image_size=32, hidden=16,
                              min_window_pixels=16, min_shift_pixels=8, max_shift_pixels=8)
    images, padding, text, reg, boxes = inputs()
    boxes[:] = torch.tensor([.5, .5, .01, .01])
    with torch.no_grad():head.offset[-1].bias.fill_(20.)
    out, _ = head(images, padding, text, reg, boxes)
    assert torch.allclose(out[:, :2] - boxes[:, :2], torch.full((2, 2), .25), atol=1e-6)
    assert torch.equal(out[:, 2:], boxes[:, 2:])
    with pytest.raises(ValueError, match='limits'):
        CenterEvidenceHead(min_shift_pixels=9, max_shift_pixels=8)


def test_center_supervision_skips_outside_window_and_all_padding():
    head = model('text_gate')
    images, padding, text, reg, boxes = inputs()
    _, aux = head(images, padding, text, reg, boxes)
    target = boxes.clone(); target[1, :2] = .01
    loss, stats = center_heatmap_loss(aux, target, return_stats=True)
    assert torch.isfinite(loss) and loss > 0 and stats['eligible_count'].item() == 1
    assert stats['eligible_fraction'].item() == .5
    loss.backward()
    assert head.rgb_projection.weight.grad.abs().sum() > 0
    assert head.tir_projection.weight.grad.abs().sum() > 0
    assert head.route[-1].weight.grad.abs().sum() > 0
    padding[:] = True
    _, empty = head(images, padding, text, reg, boxes)
    zero, stats = center_heatmap_loss(empty, target, return_stats=True)
    assert zero.item() == 0 and stats['eligible_count'].item() == 0
    assert torch.isfinite(zero)


def test_tir_detail_transfer_requires_complete_schema_and_matching_digest(tmp_path):
    head = model('fixed')
    tensor_schema = head.tir_encoder.state_dict()
    expected = {name: torch.randn_like(value) for name, value in tensor_schema.items()}
    path = tmp_path/'detail.pth'
    torch.save({'model': {'tir_detail.'+name: value for name,value in expected.items()}, 'epoch': 2}, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    before_rgb = {name: value.clone() for name, value in head.rgb_encoder.state_dict().items()}
    with pytest.raises(ValueError, match='checksum'):
        head.initialize_tir_from_detail_checkpoint(path, '0'*64)
    loaded = head.initialize_tir_from_detail_checkpoint(path, digest)
    assert loaded['sha256'] == digest and loaded['tensors'] == len(expected)
    assert all(torch.equal(head.tir_encoder.state_dict()[name], value) for name,value in expected.items())
    assert all(torch.equal(head.rgb_encoder.state_dict()[name], value) for name,value in before_rgb.items())
    incomplete = dict(expected); incomplete.pop(next(iter(incomplete)))
    torch.save({'model': {'tir_detail.'+name: value for name,value in incomplete.items()}}, path)
    with pytest.raises(ValueError, match='unique complete'):
        head.initialize_tir_from_detail_checkpoint(path)
