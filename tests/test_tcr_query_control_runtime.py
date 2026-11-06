import copy
import pytest
import torch

from models.gentle_differential import GentleDifferentialGrounder
from script_train.clip_tcr_query_control import runtime as r
from test_gentle_differential import TinyBase


@pytest.fixture
def resources(tmp_path,monkeypatch):
    base=tmp_path/'base.pth'
    base.write_bytes(b'isolated checkpoint')
    monkeypatch.setattr(r.gentle,'build',lambda args:GentleDifferentialGrounder(TinyBase(args),args))
    def initialize_student(student,args):
        assert r.sha256(args.exchange_base_checkpoint)==args.exchange_base_sha256
        r.gentle.assert_scope(student,args)
        return {'base_epoch':100,'base_sha256':args.exchange_base_sha256,'legacy_lavs_student_parameters':0}
    monkeypatch.setattr(r.gentle,'initialize',initialize_student)
    def args(case,seed=13):
        a=r.recipe(tmp_path/case,case,seed)
        a.exchange_base_checkpoint=str(base)
        a.exchange_base_sha256=r.sha256(base)
        return a
    return args


def validation(accu=.78,small=.65):
    return {'accu':accu,'accu_07':.58,'miou':.66,'accu_small':small,'miou_weak':.63,'accu_normal':.97}


@pytest.mark.parametrize('case',r.VARIANTS)
def test_recipe_mask_initialization_and_optimizer_coverage(resources,case):
    a=resources(case)
    assert not a.differential_enabled and a.epochs==40
    assert a.tcr_loss_weight==.05 and a.tcr_lr==5e-5 and a.tcr_hidden==128
    m=r.build(a)
    info=r.initialize(m,a)
    r.assert_scope(m,a)
    assert info['tcr_mode']==a.tcr_mode
    assert m.student.args is not a and m.student.args.experiment_case=='gentle_corrected'
    assert (m.tcr_target is None)==(case=='query_off')
    optimizer,_=r.optimizer_for(m,a)
    actual={g['name']:g['initial_lr'] for g in optimizer.param_groups}
    assert actual['fusion']==actual['decoder']==2e-6 and actual['visual_lora']==4e-6
    assert ('tcr' in actual)==(case!='query_off')
    if case!='query_off':
        assert actual['tcr']==5e-5
        assert m.tcr_target.bias_strength.requires_grad==(case in ('query_global','query_local'))
        assert m.tcr_target.bias_strength.item()==pytest.approx(.5)
    assert {id(p) for g in optimizer.param_groups for p in g['params']}=={id(p) for p in m.parameters() if p.requires_grad}


def test_bias_strength_is_projected_before_ema_observers(resources):
    a=resources('query_local'); m=r.build(a);r.initialize(m,a)
    opt,_=r.optimizer_for(m,a)
    observed=[]
    opt.register_step_post_hook(lambda *_: observed.append(m.tcr_target.bias_strength.item()))
    with torch.no_grad():m.tcr_target.bias_strength.fill_(-3)
    opt.step()
    assert observed[-1]==0
    with torch.no_grad():m.tcr_target.bias_strength.fill_(3)
    opt.step()
    assert observed[-1]==2


@pytest.mark.parametrize('case',r.VARIANTS)
def test_delta_restores_only_registered_scope_and_rejects_tampering_atomically(resources,case):
    a=resources(case);m=r.build(a);r.initialize(m,a)
    ck=r.delta_record(m,a,0,validation(),.78,{'test':True})
    selected={n:t.clone() for n,t in ck['model_delta'].items()}
    if case=='query_loss_only':assert 'tcr_target.bias_strength' not in selected
    if case in ('query_global','query_local'):assert 'tcr_target.bias_strength' in selected
    with torch.no_grad():
        for n,p in m.named_parameters():
            if r.selected_name(n,a):p.add_(1)
    r.restore_delta(m,a,ck)
    assert all(torch.equal(m.state_dict()[n],t) for n,t in selected.items())
    wrong=dict(ck);wrong['model_delta']=dict(ck['model_delta'])
    key=next(iter(selected));wrong['model_delta'][key]=wrong['model_delta'][key].double()
    with pytest.raises(ValueError,match='schema'):
        r.restore_delta(m,a,wrong)
    assert all(torch.equal(m.state_dict()[n],t) for n,t in selected.items())
    wrong=dict(ck);wrong['args']=copy.deepcopy(a);wrong['args'].seed=14
    with pytest.raises(ValueError,match='different'):
        r.restore_delta(m,a,wrong)


def test_gate_requires_bias_beyond_loss_and_no_head(monkeypatch):
    monkeypatch.setattr(r,'reference',lambda seed:{'validation':validation(460/608,.58)})
    no=validation(462/608,.60); loss=validation(464/608,.61); global_=validation(466/608,.62); local=validation(468/608,.63)
    runs={f'{k}_seed13':v for k,v in zip(r.VARIANTS,(no,loss,global_,local))}
    assert r.choose_promotion(runs)['passed']
    runs['query_local_seed13']=validation(467/608,.63)
    assert not r.choose_promotion(runs)['passed']
    replicated={f'{k}_seed{s}':v for s in r.SEEDS for k,v in zip(r.VARIANTS,(no,loss,global_,local))}
    assert r.replication_gate(replicated,'query_local')['passed']
    for s in (14,15):replicated[f'query_local_seed{s}']=no
    assert not r.replication_gate(replicated,'query_local')['passed']


def test_attention_effect_audit_requires_both_directions_and_restores_mode():
    from torch import nn
    from script_train.clip_tcr_query_control.preflight import measure_tcr_attention_effect
    from utils.misc import NestedTensor

    class FakeHead(nn.Module):
        def __init__(self,active):
            super().__init__()
            self.active=active
        def forward(self,exchange,inputs,output,*,apply_bias):
            rgb,tir=output
            b=rgb[0].shape[0]
            logits=torch.tensor([[0.,1.,2.,3.]]).repeat(b,1)
            pad=torch.zeros(b,4,dtype=torch.bool)
            increment=.01 if self.active else 0.
            return ([x+increment for x in rgb],[x+increment for x in tir]),{
                'rgb_logits':[logits], 'tir_logits':[logits], 'patch_padding':pad}

    class FakeModel(nn.Module):
        def __init__(self,active):
            super().__init__()
            self.tcr_target=FakeHead(active)
        def forward(self,images,phrases):
            b=images.tensors.shape[0]
            features=([torch.zeros(b,5,8)],[torch.zeros(b,5,8)])
            return self.tcr_target(None,None,features,apply_bias=True)

    images=NestedTensor(torch.zeros(2,4,8,8),torch.zeros(2,8,8,dtype=torch.bool))
    model=FakeModel(True).train()
    result=measure_tcr_attention_effect(model,images,['a','b'])
    assert model.training
    assert result['rgb_output_max_abs']>0 and result['tir_output_max_abs']>0
    assert result['rgb_prior_max_abs']>0 and result['tir_prior_max_abs']>0
    with pytest.raises(AssertionError,match='no finite effect'):
        measure_tcr_attention_effect(FakeModel(False),images,['a','b'])


def test_query_intervention_audit_checks_both_streams_and_restores_state():
    from torch import nn
    from script_train.clip_tcr_query_control.preflight import measure_query_gate_effect
    from utils.misc import NestedTensor
    class Head(nn.Module):
        def __init__(self,active):
            super().__init__()
            self.query_enabled=True
            self.active=active
        def forward(self,inputs):
            inc=.01 if self.query_enabled and self.active else 0.
            return ([inputs+inc],[inputs+inc]),{}
    class Model(nn.Module):
        def __init__(self,active):
            super().__init__()
            self.query_enabled=True
            self.tcr_target=Head(active)
        def forward(self,images,phrases):
            return self.tcr_target(images.tensors.flatten(2).transpose(1,2))
    images=NestedTensor(torch.zeros(2,4,8,8),torch.zeros(2,8,8,dtype=torch.bool))
    model=Model(True).train()
    result=measure_query_gate_effect(model,images,['a','b'])
    assert model.training and model.tcr_target.query_enabled
    assert result['rgb_query_gate_max_abs']>0 and result['tir_query_gate_max_abs']>0
    inactive=Model(False).train()
    with pytest.raises(AssertionError,match='no finite effect'):
        measure_query_gate_effect(inactive,images,['a','b'])
    assert inactive.training and inactive.tcr_target.query_enabled
