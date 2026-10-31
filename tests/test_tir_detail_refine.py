import importlib.util
from pathlib import Path

import torch
import pytest
from types import SimpleNamespace
import sys

spec = importlib.util.spec_from_file_location('tir_detail_refine', Path(__file__).parents[1] / 'models/tir_detail_refine.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
torch.set_num_threads(2)


def fixture_inputs():
    torch.manual_seed(17)
    mask = torch.zeros(2, 224, 224, dtype=torch.bool)
    mask[1, :, 160:] = True
    encoder = m.TIRDetailEncoder()
    features, masks = encoder(torch.randn(2, 1, 224, 224), mask.int())
    return encoder, features, masks


def test_encoder_local_cells_and_identity_injection():
    encoder, features, masks = fixture_inputs()
    assert [tuple(x.shape) for x in features] == [(2, 64, 56, 56), (2, 96, 28, 28), (2, 128, 14, 14)]
    injector = m.TextLocalTIRInjector()
    tokens = [torch.randn(2, 197, 768) for _ in range(4)]
    out = injector(tokens, torch.randn(2, 512), features, masks)
    for a, b in zip(tokens, out):
        assert torch.equal(a, b)
    cell_ids = torch.arange(56 * 56).reshape(1, 1, 56, 56)
    cells = m.patch_cells(cell_ids, (14, 14))
    assert cells[0, 0, :, 0].tolist() == [0, 1, 2, 3, 56, 57, 58, 59, 112, 113, 114, 115, 168, 169, 170, 171]
    sum(x.sum() for x in out).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in injector.parameters())
    assert any(p.grad.abs().sum() > 0 for p in injector.outputs.parameters())


def test_tiny_soft_region_and_padding():
    boxes = torch.tensor([[.503, .507, .001, .002], [.5, .5, .3, .3]])
    target = m.soft_box_occupancy(boxes, (56, 56))
    assert target[0].sum() > 0
    assert torch.allclose(target.sum((1, 2)), boxes[:, 2] * boxes[:, 3] * 56 ** 2, atol=2e-4)
    logits = torch.randn(2, 1, 56, 56, requires_grad=True)
    loss = m.balanced_region_bce(logits, boxes, torch.zeros(2, 56, 56, dtype=torch.bool))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    assert m.balanced_region_bce(logits, boxes, torch.ones(2, 56, 56, dtype=torch.bool)) == 0


def test_refiner_identity_tiny_and_empty_evidence():
    _, features, masks = fixture_inputs()
    refiner = m.BoundaryRefiner(True).eval()
    reg, eos, text = torch.randn(2, 512), torch.randn(2, 512), torch.randn(2, 77, 512)
    boxes = torch.tensor([[.5, .5, .2, .3], [.99, .01, 1e-5, 2e-5]])
    memories = (torch.randn(2, 512, 14, 14), torch.randn(2, 512, 14, 14))
    mmask = (torch.zeros(2, 14, 14, dtype=torch.bool),) * 2
    text_mask = torch.zeros(2, 77, dtype=torch.bool)
    out, aux = refiner(reg, eos, text, text_mask, boxes, memories, mmask, features, masks)
    assert torch.equal(out, boxes) and torch.equal(aux['boundary_b1'], boxes)
    out.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in refiner.parameters())
    with torch.no_grad():
        refiner.output[-1].bias.fill_(1)
    empty_mm = tuple(torch.ones_like(x) for x in mmask)
    empty_dm = tuple(torch.ones_like(x) for x in masks)
    out, _ = refiner(reg, eos, text, torch.ones_like(text_mask), boxes,
                     memories, empty_mm, features, empty_dm)
    assert torch.equal(out, boxes) and torch.isfinite(out).all()


def test_parameter_limits_and_full_padding_injector():
    model = torch.nn.Module()
    model.tir_detail = m.TIRDetailEncoder()
    model.tir_injector = m.TextLocalTIRInjector()
    model.tir_region_head = m.TextRegionHead()
    model.boundary_refiner = m.BoundaryRefiner(True)
    report = m.parameter_report(model)
    assert report['infrared_total'] <= 2_000_000 and report['net_added'] <= 5_000_000
    features, masks = model.tir_detail(torch.zeros(1, 1, 224, 224), torch.ones(1, 224, 224, dtype=torch.bool))
    with torch.no_grad():
        for output in model.tir_injector.outputs:
            output.bias.fill_(1)
    tokens = [torch.randn(1, 197, 768) for _ in range(4)]
    out = model.tir_injector(tokens, torch.randn(1, 512), features, masks)
    assert all(torch.equal(a, b) for a, b in zip(out, tokens))


def test_invalid_side_stays_fixed_with_learned_updates_and_bounded_boxes():
    refiner = m.BoundaryRefiner(False).eval()
    with torch.no_grad():
        refiner.output[-1].bias.fill_(-3)
    box = torch.tensor([[.02, .5, .7, .2]])
    memories = (torch.ones(1, 512, 14, 14),) * 2
    masks = (torch.zeros(1, 14, 14, dtype=torch.bool),) * 2
    out, aux = refiner(torch.zeros(1, 512), torch.zeros(1, 512), torch.zeros(1, 77, 512),
                       torch.zeros(1, 77, dtype=torch.bool), box, memories, masks)
    old_left = box[:, 0] - box[:, 2] / 2
    for b in (aux['boundary_b1'], out):
        assert torch.allclose(b[:, 0] - b[:, 2] / 2, old_left, atol=1e-7)
        assert (b >= 0).all() and (b <= 1).all() and (b[:, 2:] > 0).all()


def test_partial_padding_keeps_tiny_region_but_masks_sampled_pixels():
    mask = torch.zeros(1, 224, 224, dtype=torch.bool)
    mask[:, :, :1] = True
    features, masks = m.TIRDetailEncoder()(torch.randn(1, 1, 224, 224), mask)
    assert not masks[0][0, 20, 0]
    box = torch.tensor([[1.5 / 224, .5, .1 / 224, .1 / 224]])
    fraction = torch.nn.functional.adaptive_avg_pool2d((~mask)[:, None].float(), (56, 56))[:, 0]
    logits = torch.zeros(1, 1, 56, 56, requires_grad=True)
    m.balanced_region_bce(logits, box, masks[0], fraction).backward()
    assert logits.grad[0, 0, :, 0].min() < 0  # surviving positive region
    refiner = m.BoundaryRefiner(False).eval()
    with torch.no_grad():
        refiner.output[-1].bias.fill_(2)
    out, _ = refiner(torch.zeros(1, 512), torch.zeros(1, 512), torch.zeros(1, 77, 512),
        torch.zeros(1, 77, dtype=torch.bool), torch.tensor([[.5, .5, .2, .2]]),
        (torch.ones(1, 512, 14, 14),) * 2, (torch.zeros(1, 14, 14, dtype=torch.bool),) * 2,
        image_padding_mask=torch.ones(1, 224, 224, dtype=torch.bool))
    assert torch.equal(out, torch.tensor([[.5, .5, .2, .2]]))


def test_loss_weights_ramp_and_region_gt_stays_in_loss(monkeypatch):
    monkeypatch.setitem(sys.modules, 'models.tir_detail_refine', m)
    from utils.loss_utils import trans_vg_loss, _grounding_box_loss
    args = SimpleNamespace(use_contrastive_loss=False, use_rtcc_constrain_loss=False,
                           use_mask_loss=False, enable_tir_detail=True, enable_boundary_refiner=True)
    target = torch.tensor([[.5, .5, .2, .3]])
    b0 = torch.tensor([[.51, .52, .22, .28]], requires_grad=True)
    b1 = torch.tensor([[.50, .51, .21, .29]], requires_grad=True)
    b2 = target.clone().requires_grad_()
    logits = torch.zeros(1, 1, 56, 56, requires_grad=True)
    padding = torch.zeros(1, 56, 56, dtype=torch.bool)
    aux = {'boundary_b0': b0, 'boundary_b1': b1, 'tir_region_logits': logits, 'tir_region_padding': padding}
    first = trans_vg_loss(args, b2, target, None, None, aux=aux, epoch=0)
    fifth = trans_vg_loss(args, b2, target, None, None, aux=aux, epoch=4)
    assert torch.allclose(fifth['loss_boundary_b0'], .2 * _grounding_box_loss(b0, target, 1))
    assert torch.allclose(fifth['loss_boundary_b1'], .1 * _grounding_box_loss(b1, target, 1))
    assert torch.allclose(fifth['loss_tir_region'], .1 * m.balanced_region_bce(logits, target, padding))
    for key in ('loss_boundary_b0', 'loss_boundary_b1', 'loss_tir_region'):
        assert torch.allclose(first[key] * 5, fifth[key])
    sum(fifth.values()).backward()
    assert all(torch.isfinite(x.grad).all() for x in (b0, b1, b2, logits))
    with pytest.raises(ValueError, match='auxiliary'):
        trans_vg_loss(args, b2, target, None, None, aux=None)


def test_injector_learns_beyond_zero_heads_and_keeps_cls():
    torch.manual_seed(5)
    encoder, injector = m.TIRDetailEncoder(), m.TextLocalTIRInjector()
    tokens = [torch.randn(1, 197, 768) for _ in range(4)]
    eos, image = torch.randn(1, 512), torch.randn(1, 1, 224, 224)
    mask = torch.zeros(1, 224, 224, dtype=torch.bool)
    optimizer = torch.optim.AdamW([*encoder.parameters(), *injector.parameters()], lr=1e-3)
    for _ in range(2):
        optimizer.zero_grad()
        features, masks = encoder(image, mask)
        out = injector(tokens, eos, features, masks)
        sum(x.square().mean() for x in out).backward()
        optimizer.step()
    assert encoder.stage16[-1].net[0].weight.grad.abs().sum() > 0
    assert injector.token_query.weight.grad.abs().sum() > 0
    assert all(torch.equal(a[:, :1], b[:, :1]) for a, b in zip(tokens, out))
    assert any(not torch.equal(a[:, 1:], b[:, 1:]) for a, b in zip(tokens, out))
