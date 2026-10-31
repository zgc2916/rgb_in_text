import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from test_iafv3_equivalence import _load_fusion_module

fusion,_=_load_fusion_module()
m=__import__(fusion.__package__+'.reliable_modal_exchange',fromlist=['ReliableModalExchange'])
torch.set_num_threads(2)


def inputs():
    torch.manual_seed(8)
    rgb=[torch.randn(2,10,8) for _ in range(4)]
    tir=[torch.randn(2,10,8) for _ in range(4)]
    config=SimpleNamespace(hidden_size=8,num_attention_heads=2,attention_dropout=0.)
    rg=nn.ModuleList(fusion.CLIP_Cross_Attention(config) for _ in range(4))
    tg=copy.deepcopy(rg)
    text=torch.randn(2,6)
    padding=torch.zeros(2,9,dtype=torch.bool)
    padding[1,-3:]=True
    return rgb,tir,rg,tg,text,padding


@pytest.mark.parametrize('mode',m.MODES)
def test_all_modes_are_bitwise_c1_at_initialization(mode):
    rgb,tir,rg,tg,text,padding=inputs()
    exchange=m.ReliableModalExchange(3,8,6,4)
    out=exchange(rgb,tir,rg,tg,text,padding,mode)
    for index in range(4):
        assert torch.equal(out[0][index],rgb[index]+rg[index](rgb[index],tir[index])[0])
        assert torch.equal(out[1][index],tir[index]+tg[index](tir[index],rgb[index])[0])
    assert out[2]['exchange_correction_penalty']==0


def test_directions_have_independent_outputs():
    rgb,tir,rg,tg,text,padding=inputs()
    exchange=m.ReliableModalExchange(3,8,6,4)
    with torch.no_grad():
        exchange.gates[0].direction_heads[0].bias[0]=1.
    out=exchange(rgb,tir,rg,tg,text,padding,'direction_gate')
    assert not torch.equal(out[0][0],rgb[0]+rg[0](rgb[0],tir[0])[0])
    assert torch.equal(out[1][0],tir[0]+tg[0](tir[0],rgb[0])[0])
    assert torch.equal(out[0][0][1,-3:],(rgb[0]+rg[0](rgb[0],tir[0])[0])[1,-3:])


def test_quality_bias_downweights_damaged_donor_tokens():
    rgb,tir,rg,tg,text,padding=inputs()
    exchange=m.ReliableModalExchange(3,8,6,4)
    r,t=rgb[0],tir[0]
    zero=torch.zeros(2,10)
    damage=zero.clone()
    damage[:,1]=8.
    first=exchange._read(rg[0],r,t,text,padding,exchange.gates[0],0,zero,zero,True,False)[0]
    second=exchange._read(rg[0],r,t,text,padding,exchange.gates[0],0,zero,damage,True,False)[0]
    assert not torch.equal(first,second)
    assert torch.equal(first[1,-3:],second[1,-3:])
    all_pad=torch.ones_like(padding)
    result=exchange._read(rg[0],r,t,text,all_pad,exchange.gates[0],0,zero,damage,True,True)[0]
    assert torch.equal(result,rg[0](r,t)[0]) and torch.isfinite(result).all()


def test_corruptions_preserve_padding_other_sensor_and_rng_reproducibility():
    images=torch.full((16,4,48,48),.3)
    padding=torch.zeros(16,48,48,dtype=torch.bool)
    padding[:,-12:]=True
    torch.manual_seed(18)
    changed,labels=m.corrupt_modalities(images,padding,1.,3)
    torch.manual_seed(18)
    again,again_labels=m.corrupt_modalities(images,padding,1.,3)
    assert torch.equal(changed,again) and torch.equal(labels['damage'],again_labels['damage'])
    assert torch.equal(changed[:,:,-12:],images[:,:,-12:])
    assert torch.equal(images,torch.full_like(images,.3))
    damaged=(labels['damage'].sum(-1)>0)
    assert damaged.sum(1).eq(1).all() and damaged[:,0].any() and damaged[:,1].any()
    assert torch.equal(changed[~damaged[:,0],:3],images[~damaged[:,0],:3])
    assert torch.equal(changed[~damaged[:,1],3:],images[~damaged[:,1],3:])
    clean,targets=m.corrupt_modalities(images,padding,0.,3)
    assert torch.equal(clean,images) and not targets['damage'].any()


@pytest.mark.parametrize('mode',m.MODES)
def test_real_gradients_reach_both_direction_heads_and_quality_when_enabled(mode):
    rgb,tir,rg,tg,text,padding=inputs()
    exchange=m.ReliableModalExchange(3,8,6,4)
    optimizer=torch.optim.AdamW(exchange.parameters(),lr=.01)
    labels={'damage':torch.rand(2,2,9),'valid_fraction':(~padding).float()}
    for step in range(2):
        optimizer.zero_grad()
        out=exchange(rgb,tir,rg,tg,text,padding,mode,labels)
        loss=sum(value.square().mean() for value in out[0]+out[1])
        loss=loss+out[2].get('exchange_damage_bce',0)+.01*out[2]['exchange_correction_penalty']
        loss.backward()
        for gate in exchange.gates:
            assert all(head.weight.grad[0].abs().sum()>0 for head in gate.direction_heads)
            if mode!='direction_gate':
                assert all(head.net[-1].weight.grad.abs().sum()>0 for head in gate.damage_heads)
            if mode=='reliability_spatial':
                assert all(head.weight.grad[1:].abs().sum()>0 for head in gate.direction_heads)
            if step:
                assert gate.text_proj.weight.grad.abs().sum()>0
        optimizer.step()


def test_frozen_training_contract_survives_adapter_switch():
    model=object.__new__(fusion.MMVGFusion)
    nn.Module.__init__(model)
    model.args=SimpleNamespace(exchange_train_new_only=True,visual_exchange='reliability_gate')
    model.clip=nn.Linear(4,4)
    model.reliable_modal_exchange=m.ReliableModalExchange(3,8,6,4)
    model.other=nn.Linear(4,4)
    model.infmae_local_center=None
    def switch(_):
        for parameter in model.clip.parameters():
            parameter.requires_grad_(True)
    model.clip.set_adapter=switch
    model._configure_exchange_tuning()
    model.train()
    model._select_grounding_clip_adapter('lora_rgb')
    assert not model.training and not model.clip.training and not model.other.training
    assert model.reliable_modal_exchange.training
    assert all(not parameter.requires_grad for parameter in model.clip.parameters())
    assert all(parameter.requires_grad for parameter in model.reliable_modal_exchange.parameters())
    model.eval()
    assert not model._exchange_new_training and not model.reliable_modal_exchange.training


def test_parameter_matched_variants_and_budget():
    exchange=m.ReliableModalExchange()
    assert sum(p.numel() for p in exchange.parameters())<2_000_000
    restored=m.ReliableModalExchange()
    restored.load_state_dict(exchange.state_dict(),strict=True)
