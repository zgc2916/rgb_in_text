import importlib.util
import json
from pathlib import Path

import pytest

path=Path(__file__).parents[1]/'script_eval/clip_dense_grid_test.py'
spec=importlib.util.spec_from_file_location('dense_diagnostic_test',path)
evaluation=importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def row(index=0, **updates):
    return {'index':index,'image':'one.jpg','phrase':'person','iou':.7,**updates}


def test_last_log_alone_does_not_mean_selection_is_committed(tmp_path):
    directory=tmp_path/'explore'/'dense_336_seed13'
    directory.mkdir(parents=True)
    path=directory/'progress.json'
    path.write_text(json.dumps({'completed_epochs':20,'epoch_in_progress':True}))
    assert not evaluation.phase_one_ready(tmp_path,['dense_336_seed13'],20)
    path.write_text(json.dumps({'completed_epochs':20,'epoch_in_progress':False}))
    assert evaluation.phase_one_ready(tmp_path,['dense_336_seed13'],20)


def test_cross_resolution_gt_coordinates_can_differ_but_expression_identity_cannot():
    a=[row(target_xywh=[.5,.5,.1,.1])]
    b=[row(target_xywh=[.5,.499,.1,.1])]
    evaluation.aligned_samples({'a':a,'b':b},1)
    with pytest.raises(ValueError,match='expressions'):
        evaluation.aligned_samples({'a':a,'b':[row(phrase='car')]},1)
    with pytest.raises(ValueError,match='Incomplete'):
        evaluation.aligned_samples({'a':a},2)


def test_report_recomputes_metrics_and_paired_rescues():
    a=[row(i,iou=x) for i,x in enumerate([.5,.8,.49])]
    b=[row(i,iou=x) for i,x in enumerate([.4,.3,.8])]
    metrics=evaluation.summarize(a)
    assert metrics['total']==3 and metrics['correct_05']==2 and metrics['correct_07']==1
    changes=evaluation.paired_changes(a,b)
    assert changes['gained_hits05']==2 and changes['lost_hits05']==1 and changes['net_hits05']==1
