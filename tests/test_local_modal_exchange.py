import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

spec = importlib.util.spec_from_file_location('local_exchange', Path(__file__).parents[1] / 'models/local_modal_exchange.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
torch.set_num_threads(2)


def attention(dim=8, heads=2):
    block = nn.Module()
    block.embed_dim, block.num_heads, block.head_dim = dim, heads, dim // heads
    block.dropout = 0.
    for name in ('q_proj', 'k_proj', 'v_proj', 'out_proj'):
        setattr(block, name, nn.Linear(dim, dim))
    return block


def test_local_graph_does_not_wrap_rows_and_cls_is_global():
    graph = m.local_attention_graph(3, 1)
    assert graph[0].all() and graph[:, 0].all()
    assert graph[1, [1, 2, 4, 5]].all()
    assert not graph[3, 4]  # top right and middle left are not neighbors
    assert not graph[1, 9]


def test_read_matches_explicit_masked_attention():
    torch.manual_seed(9)
    block = attention()
    r, t = torch.randn(2, 10, 8), torch.randn(2, 10, 8)
    graph = m.local_attention_graph(3, 1)
    mask = torch.zeros(2, 9, dtype=torch.bool)
    mask[1, -3:] = True
    valid = torch.cat((torch.ones(2, 1, dtype=torch.bool), ~mask), -1)
    split = lambda x: x.reshape(2, 10, 2, 4).transpose(1, 2)
    q, k, v = split(block.q_proj(r)), split(block.k_proj(t)), split(block.v_proj(t))
    weights = ((q @ k.transpose(-1, -2)) / 2).masked_fill(~(graph[None, None] & valid[:, None, None]), -torch.inf).softmax(-1)
    expected = block.out_proj((weights @ v).transpose(1, 2).reshape(2, 10, 8)).masked_fill(~valid[:, :, None], 0)
    assert torch.allclose(m.local_cross_read(block, r, t, graph, mask), expected, atol=2e-7)


def test_distant_or_padded_tokens_cannot_enter_patch_read():
    block = attention()
    r, t = torch.randn(1, 10, 8), torch.randn(1, 10, 8)
    graph = m.local_attention_graph(3, 0)
    mask = torch.zeros(1, 9, dtype=torch.bool)
    mask[0, 3] = True
    before = m.local_cross_read(block, r, t, graph, mask)
    changed = t.clone()
    changed[:, [4, 9]] += 100
    after = m.local_cross_read(block, r, changed, graph, mask)
    assert torch.equal(before[:, 1], after[:, 1])
    assert torch.equal(before[:, 4], torch.zeros_like(before[:, 4]))
    # Padded values cannot enter even the global CLS read.
    changed = t.clone()
    changed[:, 4] += 100
    assert torch.equal(before, m.local_cross_read(block, r, changed, graph, mask))


def test_empty_evidence_stays_finite_and_out_bias_is_suppressed():
    block = attention()
    block.out_proj.bias.data.fill_(5.)
    r, t = torch.randn(2, 10, 8, requires_grad=True), torch.randn(2, 10, 8, requires_grad=True)
    result = m.local_cross_read(block, r, t, m.local_attention_graph(3, 1), torch.ones(2, 9, dtype=torch.bool))
    assert torch.isfinite(result).all() and torch.count_nonzero(result) == 0
    result.sum().backward()
    assert torch.isfinite(r.grad).all() and torch.isfinite(t.grad).all()


def test_equal_initialization_bidirectional_reads_and_checkpoint_roundtrip():
    torch.manual_seed(12)
    exchange = m.LocalModalExchange(3, (0, 1, 1, 2), 8, 6, 4)
    rgb, tir = [torch.randn(2, 10, 8) for _ in range(4)], [torch.randn(2, 10, 8) for _ in range(4)]
    reads_r, reads_t = nn.ModuleList(attention() for _ in range(4)), nn.ModuleList(attention() for _ in range(4))
    text, mask = torch.randn(2, 6), torch.zeros(2, 9, dtype=torch.bool)
    conditioned = exchange(rgb, tir, reads_r, reads_t, text, mask, True)
    plain = exchange(rgb, tir, reads_r, reads_t, text * 3, mask, False)
    for direction in range(2):
        assert all(torch.equal(a, b) for a, b in zip(conditioned[direction], plain[direction]))
    for i in range(4):
        expected = tir[i] + m.local_cross_read(reads_t[i], tir[i], rgb[i], getattr(exchange, f'graph_{i}'), mask)
        assert torch.equal(conditioned[1][i], expected)
    restored = m.LocalModalExchange(3, (0, 1, 1, 2), 8, 6, 4)
    restored.load_state_dict(exchange.state_dict(), strict=True)
    assert all(torch.equal(a, b) for a, b in zip(conditioned[0], restored(rgb, tir, reads_r, reads_t, text, mask)[0]))


def test_text_gates_learn_and_text_off_ablation_is_invariant():
    torch.manual_seed(4)
    gate = m.LocalExchangeGate(8, 6, 4)
    optimizer = torch.optim.AdamW(gate.parameters(), lr=.01)
    r, cross, text = torch.randn(2, 5, 8), torch.randn(2, 5, 8), torch.randn(2, 6)
    target = torch.rand(2, 5, 1)
    for step in range(3):
        optimizer.zero_grad()
        (gate(r, cross, text) - target).square().mean().backward()
        if step:
            assert gate.text_proj.weight.grad.abs().sum() > 0
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in gate.parameters())
        optimizer.step()
    assert not torch.equal(gate(r, cross, text), gate(r, cross, -text))
    assert torch.equal(gate(r, cross, text, False), gate(r, cross, -text, False))
    output = gate(r, cross, text)
    assert ((output > 0) & (output < 2)).all()


def test_legacy_exchange_preserves_original_math():
    from test_iafv3_equivalence import _load_fusion_module
    fusion, _ = _load_fusion_module()
    config = SimpleNamespace(hidden_size=8, num_attention_heads=2, attention_dropout=0.)
    model = object.__new__(fusion.MMVGFusion)
    nn.Module.__init__(model)
    model.visual_exchange_mode = 'lavs'
    model.cross_fusion_layers_vt = nn.ModuleList(fusion.CLIP_Cross_Attention(config) for _ in range(4))
    model.cross_fusion_layers_tv = copy.deepcopy(model.cross_fusion_layers_vt)
    rgb, tir = [torch.randn(2, 10, 8) for _ in range(4)], [torch.randn(2, 10, 8) for _ in range(4)]
    expected_r = [r + layer(r, t, output_attentions=True)[0] for r, t, layer in zip(rgb, tir, model.cross_fusion_layers_vt)]
    expected_t = [t + layer(t, r, output_attentions=True)[0] for r, t, layer in zip(rgb, tir, model.cross_fusion_layers_tv)]
    output = model._exchange_visual_levels(rgb, tir, None, None)
    assert all(torch.equal(a, b) for a, b in zip(output[0], expected_r))
    assert all(torch.equal(a, b) for a, b in zip(output[1], expected_t))


def test_rejects_invalid_graph_configuration():
    with pytest.raises(ValueError):
        m.LocalModalExchange(3, (1, 1, 1, 3), 8, 6, 4)
