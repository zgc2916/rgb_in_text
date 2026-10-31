import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

sys.path.insert(0,str(Path(__file__).parents[1]))
from script_train.clip_training_control import runtime,report,cleanup
from test_iafv3_equivalence import _load_fusion_module


def metrics(hits=476,quality=.60,miou=.67,weak=.64,epoch=4):
    return {'epoch':epoch,'accu':hits/608,'accu_07':quality,'miou':miou,'miou_weak':weak}


def test_four_controls_isolate_the_registered_factors(tmp_path):
    a,b,c,d=[vars(runtime.recipe(tmp_path,v)) for v in runtime.VARIANTS]
    diff=lambda x,y:{k for k in x if x[k]!=y[k]}
    assert diff(a,b)=={'experiment_case','use_contrastive_loss','use_rtcc_constrain_loss','use_mask_loss'}
    assert diff(b,c)=={'experiment_case','exchange_projection_tuning'}
    assert diff(c,d)=={'experiment_case','exchange_train_new_only','visual_exchange'}
    assert all(x['exchange_corruption_probability']==0 for x in [a,b,c,d])
    assert all(x['epochs']==20 and x['batch_size']*x['grad_accum_steps']*x['world_size']==96 for x in [a,b,c,d])
    assert all(x['exchange_base_sha256']==a['exchange_base_sha256'] for x in [b,c,d])
    assert d['visual_exchange']=='lavs' and not d['exchange_train_new_only']
    assert c['projection_lr']==d['projection_lr']==1e-5 and c['lr']==1e-4


@pytest.mark.parametrize('case',runtime.VARIANTS)
def test_real_model_tuning_mask_survives_peft_switch_and_mode_changes(tmp_path,case):
    fusion,_=_load_fusion_module()
    model=object.__new__(fusion.MMVGFusion)
    nn.Module.__init__(model)
    args=runtime.recipe(tmp_path,case)
    model.args=args
    model.clip=nn.Linear(4,4)
    model.cross_fusion_layers_vt=nn.ModuleList([nn.Linear(4,4)])
    model.cross_fusion_layers_tv=nn.ModuleList([nn.Linear(4,4)])
    model.other=nn.Linear(4,4)
    if case!='d_c1_continue':model.reliable_modal_exchange=nn.Linear(4,4)
    model.infmae_local_center=None
    def switch(_):
        for p in model.clip.parameters():p.requires_grad_(True)
    model.clip.set_adapter=switch
    model._configure_exchange_tuning()
    runtime.assert_scope(model,args)
    model.train()
    model._select_grounding_clip_adapter('lora_ir')
    runtime.assert_scope(model,args)
    assert not model.training and not model.clip.training and not model.other.training
    assert not model.cross_fusion_layers_vt[0].training
    if case!='d_c1_continue':assert model.reliable_modal_exchange.training
    optimizer,scheduler=runtime.optimizer_for(model,args)
    assert {id(p) for g in optimizer.param_groups for p in g['params']}=={id(p) for p in model.parameters() if p.requires_grad}
    if case=='c_joint_box':assert [g['initial_lr'] for g in optimizer.param_groups]==[1e-5,1e-4]
    model.eval()
    assert not model._exchange_new_training and not model._exchange_projection_training


@pytest.mark.parametrize('case',runtime.VARIANTS)
def test_selective_checkpoint_strictly_restores_only_its_registered_scope(tmp_path,case):
    model=nn.Module()
    model.frozen=nn.Linear(3,2)
    model.cross_fusion_layers_vt=nn.ModuleList([nn.Linear(3,2)])
    model.cross_fusion_layers_tv=nn.ModuleList([nn.Linear(3,2)])
    if case!='d_c1_continue':model.reliable_modal_exchange=nn.Linear(3,2)
    args=runtime.recipe(tmp_path,case)
    record=runtime.delta_record(model,args,0,metrics(),.8,{'test':True})
    expected={name:value.clone() for name,value in record['model_delta'].items()}
    frozen=model.frozen.weight.detach().clone()
    with torch.no_grad():
        for name,p in model.named_parameters():
            if runtime.selected_name(name,args):p.add_(10.)
    runtime.restore_delta(model,args,record)
    assert all(torch.equal(value,model.state_dict()[name]) for name,value in expected.items())
    assert torch.equal(frozen,model.frozen.weight)
    bad=copy.deepcopy(record)
    bad['model_delta']['frozen.weight']=model.frozen.weight.detach().clone()
    with pytest.raises(ValueError):runtime.restore_delta(model,args,bad)
    bad=copy.deepcopy(record)
    bad['base_sha256']='changed'
    with pytest.raises(ValueError):runtime.restore_delta(model,args,bad)


def test_continued_c1_gain_is_not_claimed_as_fusion_gain():
    base=metrics()
    runs={v:metrics(480) for v in runtime.VARIANTS}
    runs['d_c1_continue']=metrics(486)
    result=runtime.choose_promotion(base,runs)
    assert not result['joint_passed'] and result['selected_variant']=='d_c1_continue'
    assert result['replicate_variants']==['d_c1_continue']
    runs['c_joint_box']=metrics(488)
    result=runtime.choose_promotion(base,runs)
    assert result['joint_passed'] and result['replicate_variants']==['d_c1_continue','c_joint_box']
    runs['c_joint_box']=metrics(490,weak=.63)
    assert not runtime.choose_promotion(base,runs)['joint_passed']


def test_replication_requires_multiple_seed_gains_and_paired_control_gain():
    base={seed:metrics() for seed in (13,14,15)}
    candidate={seed:metrics(480) for seed in (13,14,15)}
    assert report.check_replication(candidate,base,.005)['passed']
    assert not report.check_replication(candidate,candidate,.003)['passed']
    candidate[15]=metrics(475)
    assert not report.check_replication(candidate,base,.005)['passed']


def test_zero_corruption_leaves_images_and_rng_unchanged():
    fusion,_=_load_fusion_module()
    fn=fusion.corrupt_modalities
    images=torch.randn(2,4,48,48)
    padding=torch.zeros(2,48,48,dtype=torch.bool)
    before=torch.get_rng_state().clone()
    result,targets=fn(images,padding,0.,3)
    assert result is images and not targets['damage'].any()
    assert torch.equal(before,torch.get_rng_state())


def prepare_cleanup(tmp_path,monkeypatch):
    root=tmp_path/'completed'
    kept=root/'frozen/v_seed13.pth'
    best=root/'formal/v_seed13/best_checkpoint.pth'
    kept.parent.mkdir(parents=True)
    best.parent.mkdir(parents=True)
    state={'epoch':1,'model':{'weight':torch.ones(2,2)}}
    torch.save(state,kept)
    torch.save({**state,'optimizer':{'unused':True}},best)
    cleanup.write_json(root/'status.json',{'phase':'complete','current_job':None})
    cleanup.write_json(root/'selected_models.json',[{'variant':'v','seed':13,'epoch':1,'checkpoint':str(kept),'sha256':cleanup.sha256(kept)}])
    monkeypatch.setattr(cleanup,'ORIGINAL',tmp_path)
    monkeypatch.setattr(cleanup,'ROOTS',(root,))
    return root,kept,best


def test_cleanup_preserves_frozen_model_and_rejects_changed_candidates(tmp_path,monkeypatch):
    root,kept,best=prepare_cleanup(tmp_path,monkeypatch)
    manifest=tmp_path/'audit.json'
    cleanup.audit(manifest)
    assert kept.exists() and best.exists()
    original_hash=cleanup.sha256(kept)
    state=torch.load(best,weights_only=False)
    state['model']['weight'].add_(1.)
    torch.save(state,best)
    with pytest.raises(ValueError):cleanup.execute(manifest)
    assert best.exists() and cleanup.sha256(kept)==original_hash


def test_cleanup_removes_only_verified_duplicate_after_completed_audit(tmp_path,monkeypatch):
    root,kept,best=prepare_cleanup(tmp_path,monkeypatch)
    protected=root/'other.pth'
    protected.write_bytes(b'not a cleanup candidate')
    manifest=tmp_path/'audit.json'
    cleanup.audit(manifest)
    original_hash=cleanup.sha256(kept)
    cleanup.execute(manifest)
    assert not best.exists() and protected.exists() and cleanup.sha256(kept)==original_hash
    assert json.loads(manifest.read_text())['deleted_files']==1
