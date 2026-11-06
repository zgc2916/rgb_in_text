import pytest
import torch

from models.differential_lavs import DifferentialRead
from models.gentle_differential import GentleDifferentialGrounder
from models.tcr_lavs import biased_delta, neutral_key_prior, target_alignment_loss
from models.tcr_query_lavs import QueryTCRGrounder, query_gate, query_biased_delta
from script_train.clip_gentle_differential_control import runtime as gentle
from test_gentle_differential import TinyBase
from test_tcr_lavs import _fake_base_forward, payload
from utils.misc import NestedTensor


def test_query_gate_is_neutral_on_cls_padding_and_initial_scores():
    padding=torch.tensor([[False,True,False]])
    gate=query_gate(torch.zeros(1,3),padding)
    assert torch.equal(gate,torch.ones(1,4))
    gate=query_gate(torch.tensor([[5.,-5.,-5.]]),padding)
    assert gate[0,0]==1 and gate[0,2]==1 and gate[0,1]>1 and gate[0,3]<1


def test_query_read_matches_global_for_neutral_gate_and_remains_differentiable():
    torch.manual_seed(90)
    block=DifferentialRead(8,2,6,enabled=False,preserve_foreground_context=True)
    query,source=torch.randn(2,5,8),torch.randn(2,5,8)
    prior=torch.randn(2,5,requires_grad=True)
    gate=torch.ones(2,5,requires_grad=True)
    target=query_biased_delta(block,query,source,prior,gate)
    assert torch.allclose(target,biased_delta(block,query,source,prior),atol=1e-7,rtol=1e-6)
    target.square().sum().backward()
    assert prior.grad.abs().sum()>0 and gate.grad.abs().sum()>0
    with torch.no_grad():
        gate[0,2]=0
        changed=query_biased_delta(block,query,source,prior,gate)
    assert torch.equal(changed[0,2],torch.zeros_like(changed[0,2]))
    assert torch.equal(changed[1],target.detach()[1])


def model_for(tmp_path,monkeypatch,mode,enabled):
    monkeypatch.setattr(TinyBase,'forward',_fake_base_forward,raising=False)
    args=gentle.recipe(tmp_path,'gentle_corrected',13)
    args.tcr_mode=mode
    args.tcr_hidden=4
    args.tcr_initial_strength=.5
    args.tcr_query_enabled=enabled
    torch.manual_seed(2026)
    return QueryTCRGrounder(GentleDifferentialGrounder(TinyBase(args),args),mode),args


def test_paired_models_start_identical_and_gate_has_distinct_effect(tmp_path,monkeypatch):
    off,_=model_for(tmp_path/'off',monkeypatch,'off',False)
    global_,_=model_for(tmp_path/'global',monkeypatch,'bias',False)
    query,_=model_for(tmp_path/'query',monkeypatch,'bias',True)
    inputs=payload()
    image=NestedTensor(torch.randn(2,4,8,8),torch.zeros(2,8,8,dtype=torch.bool))
    for model in (off,global_,query):model.eval()
    baseline=off(image,inputs)[0]
    assert torch.equal(baseline,global_(image,inputs)[0])
    assert torch.equal(baseline,query(image,inputs)[0])
    assert global_.tcr_target.bias_strength.item()==query.tcr_target.bias_strength.item()==.5
    # Give both controls identical nonuniform, learned-like target heads.
    torch.manual_seed(19)
    with torch.no_grad():
        for name,param in global_.tcr_target.named_parameters():
            if 'predict.2.weight' in name:param.normal_(0,.3)
        query.tcr_target.load_state_dict(global_.tcr_target.state_dict())
    captured=[]
    hooks=[model.tcr_target.register_forward_hook(lambda _,__,out: captured.append(out[0][0][0].detach()))
           for model in (global_,query)]
    global_(image,inputs)
    query(image,inputs)
    for hook in hooks:hook.remove()
    # TinyBase predicts from CLS alone, so check the patch exchange directly.
    assert torch.equal(captured[0][:,0],captured[1][:,0])
    assert not torch.equal(captured[0][:,1:],captured[1][:,1:])
    query.train()
    result=query(image,inputs)
    loss=target_alignment_loss(result[5]['tcr_target'],torch.tensor([[.3,.4,.1,.1],[.5,.5,.3,.2]]))
    (loss+result[0].square().mean()).backward()
    assert query.tcr_target.rgb.predict[-1].weight.grad.abs().sum()>0
    assert query.tcr_target.tir.predict[-1].weight.grad.abs().sum()>0
    query.args.tcr_query_enabled=False
    with pytest.raises(ValueError,match='changed'):
        query(image,inputs)
