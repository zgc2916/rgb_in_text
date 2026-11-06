import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

path=Path(__file__).parents[1]/'script_eval/clip_differential_test.py'
spec=importlib.util.spec_from_file_location('differential_diagnostic_test',path)
evaluation=importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def progress(root,name,completed=60,in_progress=False):
    p=root/'explore'/name/'progress.json'
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps({'completed_epochs':completed,'epoch_in_progress':in_progress}))


def test_test_waits_for_every_durable_training_commit_and_controller_exit(tmp_path):
    names=['lavs_control_seed13','corrected_attention_seed13','differential_attention_seed13']
    terminal={'phase':'gate_not_passed','controller_active':False}
    for name in names:progress(tmp_path,name)
    assert evaluation.training_ready(tmp_path,names,60,terminal)
    progress(tmp_path,names[-1],in_progress=True)
    assert not evaluation.training_ready(tmp_path,names,60,terminal)
    progress(tmp_path,names[-1])
    assert not evaluation.training_ready(tmp_path,names,60,dict(terminal,controller_active=True))
    assert not evaluation.training_ready(tmp_path,names,60,dict(terminal,phase='training_controls'))
    assert not evaluation.training_ready(tmp_path,names,60,dict(terminal,phase='failed',current_job=names[-1]))
    assert evaluation.training_ready(tmp_path,names,60,dict(terminal,phase='failed',current_job='evaluate_lavs_control_seed13'))


def test_completed_seed_selection_rejects_partial_replications(tmp_path):
    runtime=SimpleNamespace(SEEDS=(13,14,15),VARIANTS=('a','b','c'),EPOCHS=60)
    for name in runtime.VARIANTS:progress(tmp_path,f'{name}_seed13')
    assert evaluation.completed_seeds(tmp_path,runtime)==[13]
    progress(tmp_path,'a_seed14',completed=1,in_progress=True)
    with pytest.raises(ValueError,match='incomplete seed 14'):
        evaluation.completed_seeds(tmp_path,runtime)


def result():
    rows=[]
    for index,(box,iou) in enumerate([([.5,.5,.5,.5],1.),([.5,.5,.25,.5],.5),([0.,0.,.25,.25],0.)]):
        rows.append({'index':index,'image':f'{index}.jpg','phrase':'person',
                     'pred_xywh':box,'target_xywh':[.5,.5,.5,.5],'iou':iou})
    selected={'sha256':'abc','epoch':3,'weight_view':'ema','validation':{'accu':2/3,'accu_07':1/3,'miou':.5}}
    data={'source':{k:selected[k] for k in ['sha256','epoch','weight_view']},'split':'val',
          'summary':evaluation.summarize(rows),'samples':rows}
    return data,selected


def test_saved_predictions_recompute_at_thresholds_and_match_validation_selection():
    data,selected=result()
    check=evaluation.verify_result(data,selected,'val',3)
    assert check['passed'] and check['cpu_iou_max_error']==0.


@pytest.mark.parametrize('changed',['checkpoint','iou','summary','validation'])
def test_result_audit_rejects_tampering(changed):
    data,selected=result()
    if changed=='checkpoint':data['source']['sha256']='different'
    elif changed=='iou':data['samples'][1]['iou']=.5001
    elif changed=='summary':data['summary']['correct_05']=3
    else:selected['validation']['accu']=1.
    with pytest.raises(ValueError):evaluation.verify_result(data,selected,'val',3)


def test_registered_worker_cannot_change_after_request(tmp_path):
    worker=tmp_path/'worker.py';worker.write_text('pass\n')
    request={'training_root':str(tmp_path),'code':{
        'runner':{'path':str(path),'sha256':evaluation.sha256(path)},
        'evaluator':{'path':str(worker),'sha256':evaluation.sha256(worker)}}}
    (tmp_path/'request.json').write_text(json.dumps(request))
    evaluation.verify_request(tmp_path,tmp_path)
    worker.write_text('changed\n')
    with pytest.raises(ValueError,match='code changed'):evaluation.verify_request(tmp_path,tmp_path)


def test_original_and_frozen_checkpoint_digests_are_both_protected(tmp_path):
    root=tmp_path/'training';output=tmp_path/'test'
    root.mkdir();output.mkdir()
    (root/'source_manifest.json').write_text('{}')
    (output/'request.json').write_text('{}')
    original=root/'best.pth';original.write_bytes(b'checkpoint')
    frozen=output/'frozen.pth';frozen.write_bytes(original.read_bytes())
    plan={'request_sha256':evaluation.sha256(output/'request.json'),
          'source_manifest_sha256':evaluation.sha256(root/'source_manifest.json'),
          'annotations':{},'checkpoints':{'one':{'checkpoint':str(frozen),
            'original_checkpoint':str(original),'sha256':evaluation.sha256(original)}}}
    evaluation.verify_plan(root,output,plan)
    original.write_bytes(b'changed')
    with pytest.raises(ValueError,match='checkpoint changed'):evaluation.verify_plan(root,output,plan)
