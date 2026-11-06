from types import SimpleNamespace

import pytest
import torch
from torch import nn

from models.lavs_replacement import (WordConditioningBlock, CrossLevelDeformableExchange,
    reference_grid, sample_heads, legacy_lavs_key, replacement_key, LavsReplacementGrounder)
from script_train.clip_lavs_replacement_control import runtime as r
from test_iafv3_equivalence import _load_fusion_module


def test_sampling_uses_pixel_centers_and_has_coordinate_gradients():
    values=torch.arange(16,dtype=torch.float32).reshape(1,16,1)
    grid=reference_grid(4,values.device,values.dtype).reshape(1,16,1,1,2).requires_grad_()
    sampled=sample_heads(values,grid,4,1)
    assert torch.allclose(sampled.flatten(),values.flatten())
    sampled.sum().backward()
    assert torch.isfinite(grid.grad).all() and grid.grad.abs().sum()>0


def test_word_conditioner_masks_padding_and_really_reads_language():
    block=WordConditioningBlock(8,8)
    image=torch.randn(2,5,8)
    words=torch.randn(2,4,8)
    padding=torch.tensor([[False,False,True,True],[False,False,False,True]])
    assert torch.equal(block(image,words,padding,'rgb'),image)
    nn.init.normal_(block.output.weight,std=.1)
    before=block(image,words,padding,'rgb')
    changed=words.clone();changed[padding]+=100
    assert torch.equal(before,block(image,changed,padding,'rgb'))
    changed=words.clone();changed[:,0]*=-1
    assert not torch.equal(before,block(image,changed,padding,'rgb'))
    assert torch.equal(block(image,words,torch.ones_like(padding),'rgb'),image)


def inputs():
    torch.manual_seed(70)
    rgb=[torch.randn(2,10,12) for _ in range(2)]
    tir=[torch.randn(2,10,12) for _ in range(2)]
    words=torch.randn(2,4,6)
    wp=torch.tensor([[False,False,False,True],[False,False,True,True]])
    padding=torch.tensor([[False]*8+[True],[False]*9])
    return rgb,tir,words,wp,padding


def test_deformable_exchange_zero_init_is_the_no_lavs_visual_path():
    block=CrossLevelDeformableExchange(12,6,8,2,2)
    args=inputs()
    r,t=block(*args)
    assert all(torch.equal(a,b) for a,b in zip(r,args[0]))
    assert all(torch.equal(a,b) for a,b in zip(t,args[1]))
    (sum(x.square().mean() for x in r+t)).backward()
    assert block.outputs[0].weight.grad.abs().sum()>0


def test_offsets_get_real_gradients_and_padding_cannot_supply_messages():
    block=CrossLevelDeformableExchange(12,6,8,2,2)
    for layer in block.outputs:
        nn.init.normal_(layer.weight,std=.1)
    args=inputs()
    r,t=block(*args)
    loss=sum(x.square().mean() for x in r+t);loss.backward()
    assert sum(layer.weight.grad.abs().sum() for layer in block.offsets)>0
    changed=list(args)
    changed[1]=[x.clone() for x in args[1]]
    for x in changed[1]:x[0,-1]+=100
    after,_=block(*changed)
    assert all(torch.equal(a[0,:-1],b[0,:-1]) for a,b in zip(r,after))
    assert all(torch.equal(a[0,-1],b[0,-1]) for a,b in zip(r,args[0]))
    all_padded=[*args[:4],torch.ones_like(args[-1])]
    empty_r,empty_t=block(*all_padded)
    assert all(torch.equal(a,b) for a,b in zip(empty_r+empty_t,args[0]+args[1]))


def test_real_encoder_layer_replaces_old_text_lavs_instead_of_retaining_it():
    fusion,_=_load_fusion_module()
    class Attention(nn.Module):
        def forward(self,hidden_states,**kwargs):return hidden_states*0,None
    layer=SimpleNamespace(embed_dim=8,self_attn=Attention(),layer_norm1=nn.LayerNorm(8),
                          mlp=nn.Identity(),layer_norm2=nn.LayerNorm(8))
    args=SimpleNamespace(modality='rgbt',replace_lavs=True,replacement_dim=8)
    result=fusion.CLIPEncoderLayer_with_Crossmodal_Text_Guided_Fusion(
        args,0,layer,SimpleNamespace(layer_norm_eps=1e-5),[0],[12],SimpleNamespace(hidden_size=8))
    assert hasattr(result,'word_conditioner')
    assert not any(hasattr(result,k) for k in ['cross_attn_st','cross_attn_sv','cross_mlp_st','cross_mlp_sv'])
    words=torch.randn(2,4,8)
    value=result(torch.randn(2,5,8),0,[0],[words],'rgb',None,None,
                 language_padding_mask=torch.zeros(2,4,dtype=torch.bool))[0]
    assert value.shape==(2,5,8) and torch.isfinite(value).all()


def test_checkpoint_filter_distinguishes_removed_lavs_and_new_word_blocks():
    assert legacy_lavs_key('clip.base_model.model.vision_model.encoder.layers.0.cross_attn_sv.q_proj.lora_A.lora_rgb.weight')
    assert legacy_lavs_key('cross_fusion_layers_tv.0.q_proj.weight')
    assert not legacy_lavs_key('clip.base_model.model.vision_model.encoder.layers.0.self_attn.q_proj.base_layer.weight')
    assert replacement_key('clip.base_model.model.vision_model.encoder.layers.0.word_conditioner.query.weight')


def test_recipes_really_replace_both_parts_and_share_teacher_and_data_budget(tmp_path):
    recipes=[r.recipe(tmp_path,v) for v in r.VARIANTS]
    assert [a.replace_lavs for a in recipes]==[False,True,True]
    assert [a.visual_exchange for a in recipes]==['lavs','deformable_lavs','deformable_lavs']
    assert [a.replacement_fixed_offsets for a in recipes]==[False,True,False]
    assert all(a.imsize==224 and a.epochs==40 and a.batch_size*a.grad_accum_steps*a.world_size==96 for a in recipes)
    assert all(a.teacher_preserve_epochs==10 and a.teacher_preserve_weight==.5 for a in recipes)
    assert len({a.exchange_base_sha256 for a in recipes})==1


def test_fixed_sampling_optimizer_excludes_offset_parameters_and_base_clip(tmp_path):
    base=nn.Module()
    base.clip=nn.Linear(3,3)
    base.replacement_exchange=CrossLevelDeformableExchange(12,6,8,2,2,learn_offsets=False)
    a=r.recipe(tmp_path,'fixed_exchange')
    model=LavsReplacementGrounder(base,a)
    optimizer,_=r.optimizer_for(model,a)
    covered={id(p) for group in optimizer.param_groups for p in group['params']}
    assert not any(id(p) in covered for p in model.base.clip.parameters())
    assert not any(id(p) in covered for p in base.replacement_exchange.offsets.parameters())
    assert covered=={id(p) for p in model.parameters() if p.requires_grad}
    state=r.delta_record(model,a,0,{'accu':.8,'accu_07':.6,'miou':.67},.8,{})
    assert not any('teacher' in name or '.offsets.' in name for name in state['model_delta'])
    with pytest.raises(ValueError,match='GT boxes'):
        model(None,None,target_boxes=torch.ones(1,4))


def test_teacher_stays_outside_student_and_cannot_reopen_adapters(monkeypatch,tmp_path):
    teacher=nn.Module()
    teacher.base=nn.Module();teacher.base.clip=nn.Linear(3,3)
    monkeypatch.setattr(r,'build',lambda args:teacher)
    monkeypatch.setattr(r,'initialize',lambda *args:None)
    result=r.teacher_for(r.recipe(tmp_path),'cpu')
    assert not result.training and all(not p.requires_grad for p in result.parameters())
    assert not any(result.base._clip_control_trainability.values())


def test_continuation_gain_cannot_be_credited_to_replacement():
    base=r.reference(13)['validation']
    runs={v+'_seed13':dict(base,accu=base['accu']+6/608,accu_small=base['accu_small']+.02) for v in r.VARIANTS}
    assert not r.choose_promotion(runs)['passed']
