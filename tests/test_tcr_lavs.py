import copy
import inspect

import pytest
import torch
from torch import nn

from models.differential_lavs import DifferentialRead, DifferentialVisualExchange
from models.gentle_differential import GentleDifferentialGrounder
from models.tcr_lavs import (TargetHead, TargetKeyBias, TCRGrounder, assert_scope,
                             biased_delta, neutral_key_prior, target_alignment_loss)
from script_train.clip_gentle_differential_control import runtime as gentle
from test_gentle_differential import TinyBase
from utils.misc import NestedTensor


def test_neutral_prior_handles_cls_padding_and_constant_scores():
    scores=torch.full((2,4),-1.3)
    padding=torch.tensor([[False,False,True,False],[True,True,True,True]])
    prior=neutral_key_prior(scores,padding)
    assert prior.shape==(2,5)
    assert torch.equal(prior,torch.zeros_like(prior))
    # A realistic uneven count must also cancel bit-for-bit after centering.
    large=torch.full((2,196),-1.3)
    large_padding=torch.ones_like(large,dtype=torch.bool)
    large_padding[0,:97]=False
    large_padding[1,:20]=False
    assert torch.equal(neutral_key_prior(large,large_padding),torch.zeros(2,197))
    scores=torch.tensor([[1.,-1.,1000.,.5],[8.,9.,10.,11.]])
    prior=neutral_key_prior(scores,padding)
    assert torch.equal(prior[:,0],torch.zeros(2))
    assert prior[0,3]==0 and prior[1].count_nonzero()==0
    assert abs(prior[0,torch.tensor([1,2,4])].mean().item())<1e-7


def test_bias_zero_preserves_original_and_attention_weights_remain_trainable():
    torch.manual_seed(12)
    block=DifferentialRead(8,2,6,enabled=False,preserve_foreground_context=True)
    query,source=torch.randn(2,5,8),torch.randn(2,5,8)
    bias=torch.zeros(2,5,requires_grad=True)
    zero=biased_delta(block,query,source,bias)
    assert torch.equal(zero,torch.zeros_like(zero))
    probe=torch.randn_like(zero)
    (zero*probe).sum().backward()
    assert bias.grad is not None and bias.grad.abs().sum()>0
    with torch.no_grad():
        bias.fill_(.7)
    # An equal key bias cannot change a softmax distribution.
    assert biased_delta(block,query,source,bias).abs().max()<1e-6


def test_target_head_masks_words_and_padding_and_handles_all_padded():
    torch.manual_seed(5)
    head=TargetHead(8,6,4)
    tokens,words=torch.randn(2,5,8),torch.randn(2,3,6)
    wp=torch.tensor([[False,True,True],[True,True,True]])
    pp=torch.tensor([[False,False,True,True],[True,True,True,True]])
    expected=head(tokens,words,wp,pp)
    changed_words=words.clone();changed_words[wp]+=1000
    changed_tokens=tokens.clone();changed_tokens[:,1:][pp]-=1000
    actual=head(changed_tokens,changed_words,wp,pp)
    assert torch.equal(actual,expected)
    assert expected[pp].count_nonzero()==0 and torch.isfinite(expected).all()


def test_soft_box_loss_is_finite_balanced_and_differentiable():
    a=torch.randn(3,16,requires_grad=True)
    b=torch.randn(3,16,requires_grad=True)
    padding=torch.tensor([[False]*12+[True]*4,[False]*16,[True]*16])
    boxes=torch.tensor([[.4,.5,.04,.05],[.2,.2,.12,.15],[.5,.5,.1,.1]])
    loss=target_alignment_loss({'rgb_logits':[a],'tir_logits':[b], 'patch_padding':padding},boxes)
    assert torch.isfinite(loss) and loss.item()>0
    loss.backward()
    assert a.grad.abs().sum()>0 and b.grad.abs().sum()>0
    assert a.grad[padding].count_nonzero()==0
    assert b.grad[padding].count_nonzero()==0


def _fake_base_forward(self, image, payload):
    rgb,tir,words,word_padding,padding=payload
    rv,tv=self.differential_exchange(rgb,tir,words,word_padding,padding)
    prediction=torch.sigmoid(rv[0][:,0,:4]+tv[0][:,0,:4])
    return prediction,None,None,None,None


def make_model(tmp_path,monkeypatch,mode):
    monkeypatch.setattr(TinyBase,'forward',_fake_base_forward,raising=False)
    args=gentle.recipe(tmp_path,'gentle_corrected',13)
    args.tcr_mode=mode
    args.tcr_hidden=4
    torch.manual_seed(2026)
    student=GentleDifferentialGrounder(TinyBase(args),args)
    return TCRGrounder(student,mode),args


def payload():
    torch.manual_seed(7)
    rgb=[torch.randn(2,5,8),torch.randn(2,5,8)]
    tir=[torch.randn(2,5,8),torch.randn(2,5,8)]
    words=torch.randn(2,3,512)
    wp=torch.tensor([[False,True,True],[False,False,True]])
    pp=torch.tensor([[False,False,True,False],[False,False,False,False]])
    return rgb,tir,words,wp,pp


def test_wrapper_zero_bias_identity_train_aux_and_mode_scopes(tmp_path,monkeypatch):
    off,oa=make_model(tmp_path/'off',monkeypatch,'off')
    loss_only,la=make_model(tmp_path/'loss',monkeypatch,'loss_only')
    bias,ba=make_model(tmp_path/'bias',monkeypatch,'bias')
    inputs=payload()
    image=NestedTensor(torch.randn(2,4,8,8),torch.zeros(2,8,8,dtype=torch.bool))
    for model,args in ((off,oa),(loss_only,la),(bias,ba)):
        model.eval(); assert_scope(model,args)
    reference=off(image,inputs)[0]
    assert torch.equal(reference,loss_only(image,inputs)[0])
    assert torch.equal(reference,bias(image,inputs)[0])
    assert bias.tcr_target.bias_strength.item()==pytest.approx(.1)
    assert loss_only.tcr_target.bias_strength.item()==pytest.approx(.1)
    assert all(torch.equal(loss_only.tcr_target.state_dict()[name],value)
               for name,value in bias.tcr_target.state_dict().items())
    assert not loss_only.tcr_target.bias_strength.requires_grad
    assert bias.tcr_target.bias_strength.requires_grad
    assert 'target_boxes' not in inspect.signature(bias.forward).parameters
    bias.train()
    output=bias(image,inputs)
    assert len(output)==6 and 'tcr_target' in output[5]
    loss=target_alignment_loss(output[5]['tcr_target'],torch.tensor([[.3,.4,.1,.1],[.5,.5,.3,.2]]))
    total=loss+output[0].square().mean()
    total.backward()
    assert bias.tcr_target.rgb.predict[-1].weight.grad.abs().sum()>0
    assert bias.tcr_target.tir.predict[-1].weight.grad.abs().sum()>0
    assert bias.tcr_target.bias_strength.grad is not None
    assert torch.isfinite(bias.tcr_target.bias_strength.grad)
    assert_scope(bias,ba)
    with pytest.raises(TypeError):
        bias(image,inputs,target_boxes=torch.ones(2,4))


def test_exchange_hook_preserves_baseline_and_affects_expected_attention():
    torch.manual_seed(81)
    exchange=DifferentialVisualExchange(channels=8,heads=2,levels=1,enabled=False)
    target=TargetKeyBias(channels=8,text_channels=512,hidden=4)
    rgb,tir,words,wp,pp=payload()
    args=(rgb[:1],tir[:1],words,wp,pp)
    baseline=exchange(*args)
    (same_rgb,same_tir),aux=target(exchange,args,baseline,apply_bias=True)
    assert torch.equal(same_rgb[0],baseline[0][0])
    assert torch.equal(same_tir[0],baseline[1][0])
    assert len(aux['rgb_logits'])==len(aux['tir_logits'])==1
    with torch.no_grad():
        target.rgb.predict[-1].weight.normal_()
        target.tir.predict[-1].weight.normal_()
        target.bias_strength.fill_(1.)
    (new_rgb,new_tir),_=target(exchange,args,baseline,apply_bias=True)
    assert not torch.equal(new_rgb[0],baseline[0][0])
    assert not torch.equal(new_tir[0],baseline[1][0])
    target.bias_strength.data.fill_(-10.)
    target.project_bias_strength()
    assert target.bias_strength.item()==0


def test_supervision_breaks_uniform_prior_without_zero_strength_collapse(tmp_path,monkeypatch):
    model,args=make_model(tmp_path,monkeypatch,'bias')
    model.train()
    image=NestedTensor(torch.randn(2,4,8,8),torch.zeros(2,8,8,dtype=torch.bool))
    inputs=payload()
    boxes=torch.tensor([[.3,.4,.1,.1],[.5,.5,.3,.2]])
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-3)
    first=model(image,inputs)
    for values in (first[5]['tcr_target']['rgb_logits'],first[5]['tcr_target']['tir_logits']):
        assert all(value.count_nonzero()==0 for value in values)
    observed_strength_grad=[]
    for _ in range(2):
        optimizer.zero_grad()
        result=model(image,inputs)
        loss=result[0].square().mean()+target_alignment_loss(result[5]['tcr_target'],boxes)
        loss.backward()
        observed_strength_grad.append(float(model.tcr_target.bias_strength.grad))
        optimizer.step()
        model.tcr_target.project_bias_strength()
    assert observed_strength_grad[0]==0.
    assert observed_strength_grad[1]!=0.
    assert 0.<model.tcr_target.bias_strength.item()<=2.
    result=model(image,inputs)
    assert any(value.abs().max()>0 for value in result[5]['tcr_target']['rgb_logits'])
    assert any(value.abs().max()>0 for value in result[5]['tcr_target']['tir_logits'])
    assert_scope(model,args)
