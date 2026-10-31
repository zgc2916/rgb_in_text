import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

spec = importlib.util.spec_from_file_location('soft_exchange', Path(__file__).parents[1]/'models/soft_spatial_exchange.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
torch.set_num_threads(2)


def legacy_attention():
    from test_iafv3_equivalence import _load_fusion_module
    fusion, _ = _load_fusion_module()
    return fusion.CLIP_Cross_Attention(SimpleNamespace(hidden_size=8, num_attention_heads=2, attention_dropout=0.))


def inputs():
    torch.manual_seed(8)
    rgb, tir = [torch.randn(2, 10, 8) for _ in range(4)], [torch.randn(2, 10, 8) for _ in range(4)]
    rg = nn.ModuleList(legacy_attention() for _ in range(4))
    tg = copy.deepcopy(rg)
    text = torch.randn(2, 6)
    padding = torch.zeros(2, 9, dtype=torch.bool)
    padding[1, -3:] = True
    return rgb, tir, rg, tg, text, padding


def test_spatial_bases_are_smooth_and_leave_cls_global():
    bases = m.spatial_bases(3)
    assert bases.shape == (3, 10, 10)
    assert not bases[:, 0].any() and not bases[:, :, 0].any()
    assert torch.equal(bases[:, 1, 1], torch.ones(3))
    assert (bases[:, 1, 9] > 0).all()  # distant positions are still accessible
    assert torch.all(bases[:, 1, 2] > bases[:, 1, 9])


def test_global_projection_read_is_bitwise_legacy_equivalent():
    rgb, tir, rg, _, _, _ = inputs()
    expected = rg[0](rgb[0], tir[0], output_attentions=True)[0]
    assert torch.equal(expected, m.global_read(rg[0], rgb[0], tir[0])[0])


@pytest.mark.parametrize('use_text,use_spatial', [(False, False), (True, False), (True, True)])
def test_all_variants_start_at_legacy_including_padded_queries(use_text, use_spatial):
    rgb, tir, rg, tg, text, padding = inputs()
    exchange = m.SoftSpatialModalExchange(3, 8, 6, 4)
    out = exchange(rgb, tir, rg, tg, text, padding, use_text, use_spatial)
    for i in range(4):
        assert torch.equal(out[0][i], rgb[i] + rg[i](rgb[i], tir[i])[0])
        assert torch.equal(out[1][i], tir[i] + tg[i](tir[i], rgb[i])[0])


def test_learned_bias_changes_attention_without_discarding_far_evidence():
    rgb, tir, rg, tg, text, padding = inputs()
    exchange = m.SoftSpatialModalExchange(3, 8, 6, 4)
    for block in [*rg, *tg]:
        for proj in (block.q_proj, block.k_proj):
            nn.init.zeros_(proj.weight); nn.init.zeros_(proj.bias)
        for proj in (block.v_proj, block.out_proj):
            nn.init.eye_(proj.weight); nn.init.zeros_(proj.bias)
    recipient = torch.zeros(1, 10, 8)
    donor = torch.zeros_like(recipient); donor[:, 9] = 1.
    text = torch.zeros(1, 6); padding = torch.zeros(1, 9, dtype=torch.bool)
    with torch.no_grad():
        exchange.gates[0].output[-1].bias[1] = 1.
    global_read = exchange._read(rg[0], recipient, donor, exchange.gates[0], text, padding, True, False)
    soft_read = exchange._read(rg[0], recipient, donor, exchange.gates[0], text, padding, True, True)
    assert (soft_read[:, 1] > 0).all() and (soft_read[:, 1] < global_read[:, 1]).all()
    assert torch.equal(soft_read[:, 0], global_read[:, 0])


def test_text_and_spatial_heads_learn_beyond_zero_initialization():
    rgb, tir, rg, tg, text, padding = inputs()
    exchange = m.SoftSpatialModalExchange(3, 8, 6, 4)
    optimizer = torch.optim.AdamW(exchange.parameters(), lr=.01)
    targets = [torch.randn_like(r) for r in rgb]
    for step in range(3):
        optimizer.zero_grad()
        out = exchange(rgb, tir, rg, tg, text, padding, True, True)
        sum((a-b).square().mean() for a,b in zip(out[0], targets)).backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in exchange.parameters())
        assert exchange.gates[0].output[-1].weight.grad[1:].abs().sum() > 0
        if step:
            assert exchange.gates[0].text_proj.weight.grad.abs().sum() > 0
        optimizer.step()
    a = exchange(rgb, tir, rg, tg, text, padding, True, True)[0]
    b = exchange(rgb, tir, rg, tg, -text, padding, True, True)[0]
    assert any(not torch.equal(x,y) for x,y in zip(a,b))
    a = exchange(rgb, tir, rg, tg, text, padding, False, False)[0]
    b = exchange(rgb, tir, rg, tg, -text, padding, False, False)[0]
    assert all(torch.equal(x,y) for x,y in zip(a,b))


def test_padding_is_neutral_for_new_bias_and_gate():
    rgb, tir, rg, _, text, padding = inputs()
    exchange = m.SoftSpatialModalExchange(3, 8, 6, 4)
    with torch.no_grad():
        exchange.gates[0].output[-1].bias.fill_(2.)
    result = exchange._read(rg[0], rgb[0], tir[0], exchange.gates[0], text, padding, True, True)
    original = rg[0](rgb[0], tir[0])[0]
    assert torch.equal(result[1,-3:], original[1,-3:])
    result = exchange._read(rg[0], rgb[0], tir[0], exchange.gates[0], text, torch.ones_like(padding), True, True)
    assert torch.equal(result, original) and torch.isfinite(result).all()


def test_strict_checkpoint_and_parameter_budget():
    rgb, tir, rg, tg, text, padding = inputs()
    exchange = m.SoftSpatialModalExchange(3, 8, 6, 4)
    restored = m.SoftSpatialModalExchange(3, 8, 6, 4)
    restored.load_state_dict(exchange.state_dict(), strict=True)
    assert all(torch.equal(a,b) for a,b in zip(exchange(rgb,tir,rg,tg,text,padding)[0], restored(rgb,tir,rg,tg,text,padding)[0]))
    assert sum(p.numel() for p in m.SoftSpatialModalExchange().parameters()) == 1_331_216


def test_rejects_invalid_spatial_configuration():
    with pytest.raises(ValueError): m.spatial_bases(3, (1., -2., 3.))
    with pytest.raises(ValueError): m.SoftSpatialModalExchange(bias_bound=float('nan'))
