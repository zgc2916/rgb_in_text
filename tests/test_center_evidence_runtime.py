import copy
import inspect

import pytest
import torch

from models.center_evidence import Stride4Encoder, center_heatmap_loss
from models.gentle_differential import GentleDifferentialGrounder
from script_train.clip_center_evidence_control import runtime as r
from test_gentle_differential import TinyBase


@pytest.fixture
def resources(tmp_path, monkeypatch):
    """Use real detail initialization with small, isolated checkpoint files."""
    torch.manual_seed(51)
    expected = {name:tensor.clone() for name,tensor in Stride4Encoder(1).state_dict().items()}
    detail = tmp_path/'detail_seed13.pth'
    torch.save({'model':{'tir_detail.'+name:tensor for name,tensor in expected.items()}},detail)
    base = tmp_path/'base_c1.pth'
    base.write_bytes(b'isolated C1 initialization source')
    monkeypatch.setattr(r,'DETAIL_SOURCE',detail)
    monkeypatch.setattr(r,'DETAIL_SHA256',r.sha256(detail))
    monkeypatch.setattr(r.gentle,'build',lambda args:GentleDifferentialGrounder(TinyBase(args),args))
    def initialize_student(student,args):
        assert r.sha256(args.exchange_base_checkpoint)==args.exchange_base_sha256
        r.gentle.assert_scope(student,args)
        return {'base_epoch':100,'base_sha256':args.exchange_base_sha256,
                'legacy_lavs_student_parameters':0,'migration':{'foreground_tensors_exact':True}}
    monkeypatch.setattr(r.gentle,'initialize',initialize_student)
    def make_args(case,seed=13):
        args = r.recipe(tmp_path/case,case,seed)
        args.exchange_base_checkpoint = str(base)
        args.exchange_base_sha256 = r.sha256(base)
        return args
    return {'args':make_args,'detail':detail,'base':base,'expected':expected}


def validation(accu=.78,small=.65):
    return {'accu':accu,'accu_07':.58,'miou':.66,'accu_small':small,
            'miou_weak':.63,'accu_normal':.97}


def make_model(resources,case='center_fixed',seed=13):
    args = resources['args'](case,seed)
    model = r.build(args)
    info = r.initialize(model,args)
    return model,args,info


def test_recipes_are_matched_and_use_one_declared_detail_source(resources):
    args = [resources['args'](case) for case in r.VARIANTS]
    assert [a.center_evidence_enabled for a in args] == [False,True,True]
    assert [a.center_evidence_mode for a in args] == ['fixed','fixed','text_gate']
    assert all(not a.center_evidence_freeze_student and a.center_evidence_loss_weight==.05 for a in args)
    assert all(a.epochs==40 and a.imsize==224 and a.batch_size*a.grad_accum_steps*a.world_size==96 for a in args)
    assert all(a.ema_decay==.99 and a.teacher_preserve_epochs==0 for a in args)
    assert all(a.projection_lr==2e-6 and a.late_visual_lora_lr==4e-6 and a.replacement_lr==2e-5 for a in args)
    assert all(a.center_evidence_lr==5e-5 for a in args)
    assert len({a.exchange_base_sha256 for a in args})==1
    assert len({a.center_evidence_detail_sha256 for a in args})==1
    seed14=resources['args']('center_fixed',14)
    assert seed14.center_evidence_detail_checkpoint==args[1].center_evidence_detail_checkpoint
    assert seed14.center_evidence_detail_sha256==args[1].center_evidence_detail_sha256
    assert r.NEW_PREFIX=='center_head.' and all(name.startswith('student.base.') for name in r.BASE_PREFIXES)
    assert r.ORDER[0]=='center_textgate'
    with pytest.raises(ValueError,match='Unregistered'):
        r.recipe('/tmp','unknown',13)


@pytest.mark.parametrize('case',r.VARIANTS)
def test_student_is_a_separately_copied_gentle_differential_configuration(resources,case):
    args=resources['args'](case)
    model=r.build(args)
    assert model.args is model.student.args and model.args is not args
    assert model.args.experiment_case=='gentle_differential' and args.experiment_case==case
    assert (model.center_head is not None)==(case!='center_none')
    assert model.student._gentle_contract['experiment_case']=='gentle_differential'
    r.assert_scope(model,args)
    # Coordinated mutation of both argument copies must not relabel the model.
    other='center_textgate' if case!='center_textgate' else 'center_fixed'
    args.experiment_case=other
    args.center_evidence_enabled=True
    args.center_evidence_mode='text_gate' if other=='center_textgate' else 'fixed'
    model.student.args.center_evidence_enabled=args.center_evidence_enabled
    model.student.args.center_evidence_mode=args.center_evidence_mode
    with pytest.raises(ValueError,match='different experiment'):
        r.assert_scope(model,args)


@pytest.mark.parametrize('case', ['center_fixed','center_textgate'])
def test_detail_initialization_exact_and_source_metadata_shared(resources,case):
    model,args,info=make_model(resources,case)
    assert info['detail_source_seed']==13 and info['detail_source_shared_across_seeds']
    assert info['detail_initialization']['sha256']==args.center_evidence_detail_sha256
    assert info['detail_initialization']['source_prefix']=='tir_detail.'
    assert all(torch.equal(model.center_head.tir_encoder.state_dict()[name],value)
               for name,value in resources['expected'].items())
    assert info['legacy_lavs_student_parameters']==0


@pytest.mark.parametrize('case',r.VARIANTS)
def test_optimizer_covers_only_registered_student_and_center_parameters(resources,case):
    model,args,_=make_model(resources,case)
    r.assert_scope(model,args)
    optimizer,_=r.optimizer_for(model,args)
    actual={group['name']:group['initial_lr'] for group in optimizer.param_groups}
    assert actual['decoder']==actual['fusion']==2e-6
    assert actual['visual_lora']==4e-6 and actual['differential']==2e-5
    assert ('center' in actual)==(case!='center_none')
    if case!='center_none':assert actual['center']==5e-5
    assert {id(p) for group in optimizer.param_groups for p in group['params']}=={
        id(p) for p in model.parameters() if p.requires_grad}
    params=dict(model.named_parameters())
    assert params['student.base.differential_exchange.rgb.0.query.weight'].requires_grad
    assert params['student.base.differential_exchange.rgb.0.reference_query.weight'].requires_grad
    assert not params['student.base.clip.vision_model.encoder.layers.3.language_replacement.branches.rgb.attention.query.weight'].requires_grad
    assert not params['student.base.clip.text_model.weight'].requires_grad
    if case!='center_none':
        gate=[p.requires_grad for n,p in model.named_parameters() if n.startswith('center_head.route.')]
        assert gate and all(value==(case=='center_textgate') for value in gate)
        assert r.parameter_group('center_head.route.0.weight',args)==('center' if case=='center_textgate' else None)


def test_raw_gt_cannot_be_passed_to_forward(resources):
    model,_,_=make_model(resources,'center_fixed')
    assert 'target_boxes' not in inspect.signature(model.forward).parameters
    with pytest.raises(TypeError):
        model(None,None,target_boxes=torch.ones(1,4))


@pytest.mark.parametrize('case',['center_fixed','center_textgate'])
def test_registered_optimizer_receives_real_center_heatmap_gradients(resources,case):
    model,args,_=make_model(resources,case)
    optimizer,_=r.optimizer_for(model,args)
    model.train()
    images=torch.randn(2,4,224,224)
    padding=torch.zeros(2,224,224,dtype=torch.bool)
    text,reg=torch.randn(2,512),torch.randn(2,512)
    coarse=torch.tensor([[.5,.5,.12,.15],[.4,.6,.07,.08]])
    targets=torch.tensor([[.51,.52,.12,.15],[.41,.59,.07,.08]])
    output,aux=model.center_head(images,padding,text,reg,coarse)
    assert torch.equal(output,coarse)
    loss=(output-targets).abs().sum()+args.center_evidence_loss_weight*center_heatmap_loss(aux,targets)
    optimizer.zero_grad();loss.backward()
    assert model.center_head.rgb_projection.weight.grad.abs().sum()>0
    assert model.center_head.tir_projection.weight.grad.abs().sum()>0
    if case=='center_fixed':
        assert all(p.grad is None for p in model.center_head.route.parameters())
    else:
        assert model.center_head.route[-1].weight.grad.abs().sum()>0
    before=model.center_head.offset[-1].weight.detach().clone()
    optimizer.step()
    assert not torch.equal(before,model.center_head.offset[-1].weight)
    r.assert_scope(model,args)


def test_delta_restores_all_selected_weights_and_never_saves_fixed_gate(resources):
    model,args,_=make_model(resources,'center_fixed')
    record=r.delta_record(model,args,0,validation(),.78,{'test':True})
    assert record['experiment_case']=='center_fixed' and record['detail_source_seed']==13
    assert any(name.startswith('student.base.') for name in record['model_delta'])
    assert any(name.startswith('center_head.') for name in record['model_delta'])
    assert not any(name.startswith('center_head.route.') for name in record['model_delta'])
    expected={name:value.clone() for name,value in record['model_delta'].items()}
    with torch.no_grad():
        for name,p in model.named_parameters():
            if r.selected_name(name,args):p.add_(1.)
    r.restore_delta(model,args,record)
    assert all(torch.equal(model.state_dict()[name],value) for name,value in expected.items())
    wrong=dict(record);wrong['model_delta']=dict(record['model_delta'])
    key=next(iter(wrong['model_delta']))
    wrong['model_delta'][key]=wrong['model_delta'][key].double()
    with pytest.raises(ValueError,match='tensor schema'):
        r.restore_delta(model,args,wrong)
    assert all(torch.equal(model.state_dict()[name],value) for name,value in expected.items())
    wrong=dict(record);wrong['model_delta']=dict(record['model_delta'])
    route=next(name for name in model.state_dict() if name.startswith('center_head.route.'))
    wrong['model_delta'][route]=model.state_dict()[route].clone()
    with pytest.raises(ValueError,match='delta scope'):
        r.restore_delta(model,args,wrong)


@pytest.mark.parametrize('field',['format','experiment_case','base_sha256','detail_sha256','detail_checkpoint','detail_source_seed','seed'])
def test_restore_rejects_wrong_case_seed_format_or_source(resources,field):
    model,args,_=make_model(resources,'center_textgate')
    record=r.delta_record(model,args,0,validation(),.78,{})
    wrong=dict(record)
    if field=='seed':
        wrong['args']=copy.deepcopy(record['args']);wrong['args'].seed=14
    else:
        wrong[field]='tampered'
    with pytest.raises(ValueError,match='different center'):
        r.restore_delta(model,args,wrong)


@pytest.mark.parametrize('source',['base','detail'])
def test_restore_detects_physical_initialization_source_changes(resources,source):
    model,args,_=make_model(resources,'center_fixed')
    record=r.delta_record(model,args,0,validation(),.78,{})
    resources[source].write_bytes(b'changed source')
    with pytest.raises(ValueError,match='source checksum'):
        r.restore_delta(model,args,record)


def test_training_requires_verified_initialization_and_variant_matches(resources):
    args=resources['args']('center_fixed')
    model=r.build(args)
    with pytest.raises(ValueError,match='initialized'):
        r.optimizer_for(model,args)
    with pytest.raises(ValueError,match='initialized'):
        r.delta_record(model,args,0,validation(),.78,{})
    wrong=resources['args']('center_textgate')
    with pytest.raises(ValueError,match='different experiment'):
        r.initialize(model,wrong)


def test_validation_gate_attribution_and_replication(monkeypatch):
    base=validation(460/608,.58)
    monkeypatch.setattr(r,'reference',lambda seed:{'validation':base})
    none=validation(462/608,.60)
    fixed=validation(464/608,.61)
    text=validation(466/608,.62)
    runs={'center_none_seed13':none,'center_fixed_seed13':fixed,'center_textgate_seed13':text}
    decision=r.choose_promotion(runs)
    assert decision['passed'] and decision['selected_variant']=='center_textgate'
    runs['center_textgate_seed13']=validation(465/608,.62)
    decision=r.choose_promotion(runs)
    assert decision['center_fixed_passed'] and not decision['center_textgate_passed']
    assert decision['selected_variant']=='center_fixed'
    runs['center_fixed_seed13']=dict(fixed,miou_weak=.62)
    assert not r.choose_promotion(runs)['passed']
    replicated={}
    for seed in r.SEEDS:
        replicated[f'center_none_seed{seed}']=none
        replicated[f'center_fixed_seed{seed}']=fixed
        replicated[f'center_textgate_seed{seed}']=text
    assert r.replication_gate(replicated,'center_textgate')['passed']
    for seed in (14,15):replicated[f'center_textgate_seed{seed}']=none
    assert not r.replication_gate(replicated,'center_textgate')['passed']
    with pytest.raises(ValueError,match='registered'):
        r.replication_gate(replicated,'center_none')


def test_training_loop_receives_center_runtime_explicitly(monkeypatch,tmp_path):
    seen={}
    def shared(root,directory,variant,seed,resume=False,runtime=None):
        seen.update(root=root,directory=directory,variant=variant,seed=seed,resume=resume,runtime=runtime)
        return 'called'
    monkeypatch.setattr(r.common,'train',shared)
    assert r.train(tmp_path,tmp_path/'run','center_textgate',13,True)=='called'
    assert seen=={'root':tmp_path,'directory':tmp_path/'run','variant':'center_textgate',
                  'seed':13,'resume':True,'runtime':r}
