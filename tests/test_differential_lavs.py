from types import SimpleNamespace

import pytest
import torch
from torch import nn

from models.differential_lavs import (
    DifferentialRead, DifferentialLanguageBlock, DifferentialVisualExchange,
    DifferentialGrounder, migrate_lavs_state, contrast_key, selected_name,
)
from script_train.clip_differential_control import runtime as r
from test_iafv3_equivalence import _load_fusion_module


def transfer_read(block, legacy):
    for new, old in [('query','q_proj'),('key','k_proj'),('value','v_proj'),('output','out_proj')]:
        getattr(block,new).load_state_dict(getattr(legacy,old).state_dict())


def test_corrected_read_matches_independent_original_and_exposes_old_batch_mixing():
    fusion,_=_load_fusion_module()
    torch.manual_seed(13)
    old=fusion.CLIP_Cross_Attention_VS(SimpleNamespace(hidden_size=8,num_attention_heads=2,attention_dropout=0.))
    new=DifferentialRead(8,2,8,enabled=False)
    transfer_read(new,old)
    visual,words=torch.randn(3,5,8),torch.randn(3,4,8)
    expected=torch.cat([old(visual[i:i+1],words[i:i+1].transpose(0,1))[0] for i in range(3)])
    actual=new(words,visual,context=words,roundtrip=True)
    assert torch.allclose(actual,expected,atol=1e-7)
    old_batch=old(visual,words.transpose(0,1))[0]
    assert not torch.allclose(actual,old_batch,atol=1e-4)
    changed=words.clone();changed[1]+=10
    assert torch.equal(actual[0],new(changed,visual,context=changed,roundtrip=True)[0])
    order=torch.tensor([2,0,1]);inverse=torch.argsort(order)
    assert torch.equal(actual,new(words[order],visual[order],context=words[order],roundtrip=True)[inverse])


def test_full_width_cross_attention_migrates_without_initial_function_change():
    fusion,_=_load_fusion_module()
    old=fusion.CLIP_Cross_Attention(SimpleNamespace(hidden_size=8,num_attention_heads=2,attention_dropout=0.))
    new=DifferentialRead(8,2,6)
    transfer_read(new,old)
    q,source,words=torch.randn(2,5,8),torch.randn(2,7,8),torch.randn(2,4,6)
    assert torch.allclose(new(q,source,context=words),old(q,source)[0],atol=1e-7)


def test_migrated_foreground_keeps_clip_context_while_reference_excludes_padding():
    # Subtracting foreground outputs after a large padding perturbation needs
    # float64 to isolate the small reference update from cancellation error.
    block=DifferentialRead(8,2,6,preserve_foreground_context=True).double()
    q,source,words=torch.randn(2,4,8,dtype=torch.float64),torch.randn(2,6,8,dtype=torch.float64),torch.randn(2,5,6,dtype=torch.float64)
    qp=torch.tensor([[False,False,True,True],[False]*4])
    sp=torch.tensor([[False,False,False,True,True,True],[False]*6])
    wp=torch.tensor([[False,False,True,True,True],[False]*5])
    kwargs=dict(context=words,context_padding=wp,query_padding=qp,source_padding=sp)
    initial=block(q,source,**kwargs)
    block.enabled=False
    assert torch.equal(initial,block(q,source,context=words))
    with torch.no_grad():block.head_coefficient.fill_(.5)
    def reference_contribution():
        block.enabled=False;foreground=block(q,source,**kwargs)
        block.enabled=True
        return block(q,source,**kwargs)-foreground
    before=reference_contribution()
    source[sp]+=100;words[wp]-=300;q[qp]+=100
    assert torch.allclose(before,reference_contribution(),atol=1e-10)


@pytest.mark.parametrize('roundtrip',[False,True])
def test_padding_cannot_supply_foreground_reference_or_routing_evidence(roundtrip):
    block=DifferentialRead(8,2,6)
    with torch.no_grad():
        block.head_coefficient.fill_(.5)
        block.coefficient_route.weight.normal_(std=.2)
    q,source,words=torch.randn(2,4,8),torch.randn(2,6,8),torch.randn(2,5,6)
    qp=torch.tensor([[False,False,True,True],[False]*4])
    sp=torch.tensor([[False,False,False,True,True,True],[True]*6])
    wp=torch.tensor([[False,False,True,True,True],[True]*5])
    kwargs=dict(context=words,context_padding=wp,query_padding=qp,source_padding=sp,roundtrip=roundtrip)
    expected=block(q,source,**kwargs)
    source[sp]+=100;words[wp]-=300;q[qp]+=100
    assert torch.equal(expected,block(q,source,**kwargs))
    assert torch.isfinite(expected).all() and expected[1].count_nonzero()==0


def test_real_language_layer_has_new_parameters_and_no_original_lavs():
    fusion,_=_load_fusion_module()
    class Attention(nn.Module):
        def forward(self,hidden_states,**kwargs):return hidden_states*0,None
    legacy=SimpleNamespace(embed_dim=8,self_attn=Attention(),layer_norm1=nn.LayerNorm(8),
                           mlp=nn.Identity(),layer_norm2=nn.LayerNorm(8))
    args=SimpleNamespace(modality='rgbt',replace_lavs=True,visual_exchange='differential_lavs',differential_enabled=True)
    config=SimpleNamespace(layer_norm_eps=1e-5,num_attention_heads=2,intermediate_size=16)
    layer=fusion.CLIPEncoderLayer_with_Crossmodal_Text_Guided_Fusion(args,0,legacy,config,[0],[12],SimpleNamespace(hidden_size=8))
    assert hasattr(layer,'language_replacement')
    assert not any(hasattr(layer,k) for k in ['cross_attn_st','cross_attn_sv','cross_mlp_st','cross_norm_sv','word_conditioner'])
    words=torch.randn(2,4,8)
    value=layer(torch.randn(2,5,8),0,[0],[words],'rgb',None,None,
                language_padding_mask=torch.zeros(2,4,dtype=torch.bool))[0]
    assert value.shape==(2,5,8) and torch.isfinite(value).all()


def tiny_base():
    base=nn.Module();base.clip=nn.Module();base.clip.layers=nn.ModuleList([nn.Module()])
    base.clip.layers[0].language_replacement=DifferentialLanguageBlock(8,2,16)
    base.differential_exchange=DifferentialVisualExchange(8,2,levels=2)
    return base


def original_for(base):
    result={}
    for name,tensor in base.state_dict().items():
        if contrast_key(name):continue
        old=name
        if '.language_replacement.branches.' in name:
            prefix,rest=name.split('.language_replacement.branches.')
            modality,rest=rest.split('.',1);suffix='sv' if modality=='rgb' else 'st'
            if rest.startswith('norm.'):old=prefix+'.cross_norm_'+suffix+'.'+rest[5:]
            elif rest.startswith('expand.'):old=prefix+'.cross_mlp_'+suffix+'.fc1.'+rest[7:]
            elif rest.startswith('contract.'):old=prefix+'.cross_mlp_'+suffix+'.fc2.'+rest[9:]
            else:old=prefix+'.cross_attn_'+suffix+'.'+rest[len('attention.'):]
        elif name.startswith('differential_exchange.'):
            _,modality,level,rest=name.split('.',3)
            old=f'cross_fusion_layers_{"vt" if modality=="rgb" else "tv"}.{level}.{rest}'
        for new,legacy in [('query','q_proj'),('key','k_proj'),('value','v_proj'),('output','out_proj')]:
            old=old.replace('.'+new+'.','.'+legacy+'.')
        result[old]=torch.randn_like(tensor)
    return result


def test_migration_covers_every_foreground_tensor_and_rejects_incomplete_source():
    base=tiny_base();original=original_for(base)
    contrast={n:t.clone() for n,t in base.state_dict().items() if contrast_key(n)}
    info=migrate_lavs_state(base,original)
    assert info['foreground_tensors_exact'] and info['migrated_tensors']==len(original)
    assert not info['historical_batch_mixing_preserved']
    assert all(torch.equal(base.state_dict()[n],v) for n,v in contrast.items())
    original.pop(next(iter(original)))
    with pytest.raises(ValueError,match='Missing migration key'):migrate_lavs_state(base,original)


def test_differential_reference_and_language_routing_receive_real_updates():
    block=DifferentialRead(8,2,6)
    q,source,words=torch.randn(2,4,8),torch.randn(2,6,8),torch.randn(2,5,6)
    baseline=block(q,source,context=words)
    optimizer=torch.optim.AdamW(block.parameters(),lr=.01)
    before=block.reference_query.weight.detach().clone()
    for _ in range(2):
        optimizer.zero_grad();block(q,source,context=words).square().sum().backward();optimizer.step()
    assert block.head_coefficient.grad.abs().sum()>0
    assert block.coefficient_route.weight.grad.abs().sum()>0
    assert block.reference_query.weight.grad.abs().sum()>0
    assert not torch.equal(block.reference_query.weight,before)
    assert not torch.equal(block(q,source,context=words),baseline)
    assert not torch.equal(block(q,source,context=words),block(q,source,context=-words))


def test_corrected_only_keeps_contrast_frozen_and_delta_scope_strict(tmp_path):
    args=r.recipe(tmp_path,'corrected_attention')
    model=DifferentialGrounder(tiny_base(),args)
    assert all(not p.requires_grad for n,p in model.named_parameters() if contrast_key(n))
    optimizer,_=r.optimizer_for(model,args)
    assert {id(p) for g in optimizer.param_groups for p in g['params']}=={id(p) for p in model.parameters() if p.requires_grad}
    record=r.delta_record(model,args,0,dict(accu=.5,accu_07=.3,miou=.4),.5,{})
    assert all(not contrast_key(n) for n in record['model_delta'])
    expected={n:t.clone() for n,t in record['model_delta'].items()}
    with torch.no_grad():
        for n,p in model.named_parameters():
            if selected_name(n,args):p.add_(1.)
    r.restore_delta(model,args,record)
    assert all(torch.equal(model.state_dict()[n],v) for n,v in expected.items())
    record['model_delta']['base.unregistered.weight']=torch.ones(1)
    with pytest.raises(ValueError,match='scope'):r.restore_delta(model,args,record)


def test_recipes_share_data_budget_and_new_model_runs_first(tmp_path):
    args=[r.recipe(tmp_path,v) for v in r.VARIANTS]
    assert [a.replace_lavs for a in args]==[False,True,True]
    assert [a.differential_enabled for a in args]==[False,False,True]
    assert all(a.epochs==60 and a.imsize==224 and a.batch_size*a.grad_accum_steps*a.world_size==96 for a in args)
    assert all(a.teacher_preserve_epochs==0 and a.teacher_preserve_weight==0 for a in args)
    assert r.ORDER[0]=='differential_attention'
    assert len({a.exchange_base_sha256 for a in args})==1


def test_continuation_or_layout_fix_alone_cannot_be_credited_to_differential_attention():
    baseline=r.reference(13)['validation']
    improved=dict(baseline,accu=baseline['accu']+6/608,accu_small=baseline['accu_small']+.02)
    runs={v+'_seed13':dict(improved) for v in r.VARIANTS}
    assert not r.choose_promotion(runs)['passed']
    runs['lavs_control_seed13']=baseline
    decision=r.choose_promotion(runs)
    assert decision['corrected_attention_passed'] and not decision['differential_attention_passed']
    assert decision['selected_variant']=='corrected_attention'
