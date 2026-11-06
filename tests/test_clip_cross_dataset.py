import copy
from pathlib import Path

import pytest
import torch

from models.gentle_differential import GentleDifferentialGrounder
from script_train.clip_cross_dataset import runtime as r
from test_gentle_differential import TinyBase


def validation():
    return {'accu':.7,'accu_07':.5,'miou':.6}


@pytest.fixture
def resources(tmp_path,monkeypatch):
    base=tmp_path/'c1.pth'
    base.write_bytes(b'isolated c1 base')
    monkeypatch.setattr(r.query.gentle,'build',lambda a:GentleDifferentialGrounder(TinyBase(a),a))
    def initialize_student(student,args):
        assert r.sha256(args.exchange_base_checkpoint)==args.exchange_base_sha256
        r.query.gentle.assert_scope(student,args)
        return {'base_epoch':100,'base_sha256':args.exchange_base_sha256,'legacy_lavs_student_parameters':0}
    monkeypatch.setattr(r.query.gentle,'initialize',initialize_student)
    original_recipe=r.query.recipe
    def recipe(directory,variant='query_local',seed=13):
        a=original_recipe(directory,variant,seed)
        a.exchange_base_checkpoint=str(base)
        a.exchange_base_sha256=r.sha256(base)
        return a
    monkeypatch.setattr(r.query,'recipe',recipe)
    source_args=recipe(tmp_path/'source','query_local',13)
    source=r.query.build(source_args)
    r.query.initialize(source,source_args)
    record=r.query.delta_record(source,source_args,3,validation(),.7,{'source':True})
    record['weight_view']='ema'
    source_path=tmp_path/'source.pth'
    torch.save(record,source_path)
    monkeypatch.setattr(r,'SOURCE',source_path)
    monkeypatch.setattr(r,'SOURCE_SHA256',r.sha256(source_path))
    return source_path


@pytest.mark.parametrize('case,dataset',[('cross_m3fd','rgbtvg_m3fd'),('cross_mfad','rgbtvg_mfad')])
def test_independent_dataset_contract_and_source_initialization(resources,tmp_path,case,dataset):
    args=r.recipe(tmp_path/case,case,13)
    assert args.dataset==dataset and args.epochs==40 and args.tcr_query_enabled
    model=r.build(args)
    info=r.initialize(model,args)
    assert info['source_dataset']=='rgbtvg_flir' and info['target_dataset']==dataset
    assert model.student.args.dataset==dataset
    assert model.student.args.experiment_case=='gentle_corrected'
    r.assert_scope(model,args)
    opt,_=r.optimizer_for(model,args)
    assert {id(p) for g in opt.param_groups for p in g['params']}=={id(p) for p in model.parameters() if p.requires_grad}
    wrong=copy.deepcopy(args)
    wrong.dataset='rgbtvg_flir'
    with pytest.raises(ValueError,match='dataset'):
        r.assert_scope(model,wrong)


@pytest.mark.parametrize('case',r.VARIANTS)
def test_resume_delta_round_trip_and_wrong_dataset_rejected(resources,tmp_path,case):
    args=r.recipe(tmp_path/case,case,13)
    model=r.build(args)
    r.initialize(model,args)
    record=r.delta_record(model,args,0,validation(),.7,{'experiment':True})
    selected={n:t.clone() for n,t in record['model_delta'].items()}
    with torch.no_grad():
        for name,p in model.named_parameters():
            if r.selected_name(name,args):p.add_(1)
    r.restore_delta(model,args,record)
    assert all(torch.equal(model.state_dict()[n],t) for n,t in selected.items())
    changed=dict(record)
    changed['source_sha256']='bad'
    with pytest.raises(ValueError,match='dataset or initialization'):
        r.restore_delta(model,args,changed)
    tampered=dict(record)
    tampered['model_delta']=dict(record['model_delta'])
    key=next(iter(tampered['model_delta']))
    tampered['model_delta'][key]=tampered['model_delta'][key].double()
    with pytest.raises(ValueError,match='schema'):
        r.restore_delta(model,args,tampered)
    assert all(torch.equal(model.state_dict()[n],t) for n,t in selected.items())
