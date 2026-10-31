import json
from pathlib import Path
import sys

import pytest
import torch
from torch import nn

sys.path.insert(0,str(Path(__file__).parents[1]))
from script_train.clip_reliability_control import runtime,report


def metrics(hits=476,quality=.60,miou=.67,weak=.64,epoch=4):
    return {'epoch':epoch,'accu':hits/608,'accu_07':quality,'miou':miou,'miou_weak':weak}


def test_recipe_is_matched_and_uses_seed_matched_trained_c1(tmp_path):
    recipes=[vars(runtime.recipe(tmp_path,v)) for v in runtime.VARIANTS]
    for a in recipes[1:]:
        assert {k for k in a if a[k]!=recipes[0][k]}=={'visual_exchange'}
    a=recipes[0]
    assert a['batch_size']*a['grad_accum_steps']*a['world_size']==96
    assert a['epochs']==20 and a['exchange_train_new_only']
    assert a['exchange_corruption_probability']==.35
    assert 'c1_seed13.pth' in a['exchange_base_checkpoint']
    assert 'c1_seed14.pth' in runtime.recipe(tmp_path,seed=14).exchange_base_checkpoint
    assert not a['enable_tir_detail'] and not a['enable_boundary_refiner']


def test_selection_keeps_initial_c1_on_ties_or_all_worse_epochs(tmp_path):
    runtime.write_json(tmp_path/'initial_validation.json',{'accu':.8})
    rows=[{'epoch':i,'val_accu':.8 if i==4 else .7} for i in range(20)]
    (tmp_path/'log.txt').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    runtime.write_json(tmp_path/'best.json',{'epoch':-1,'accu':.8})
    assert runtime.best_from_log(tmp_path)['epoch']==-1
    runtime.write_json(tmp_path/'best.json',{'epoch':4,'accu':.8})
    with pytest.raises(ValueError):
        runtime.best_from_log(tmp_path)


def test_promotion_needs_overall_and_weak_group_quality(tmp_path):
    base=metrics()
    runs={v:metrics(480) for v in runtime.VARIANTS}
    runs['reliability_gate']=metrics(482,weak=.63)
    runs['reliability_spatial']=metrics(481)
    assert runtime.choose_promotion(base,runs)['selected_variant']=='reliability_spatial'
    runs['reliability_spatial']=metrics(481,quality=.59)
    assert runtime.choose_promotion(base,runs)['selected_variant']=='direction_gate'
    runs['direction_gate']=metrics(479)
    assert runtime.choose_promotion(base,runs)['selected_variant'] is None


def test_compact_checkpoint_is_strictly_bound_to_c1(tmp_path):
    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.reliable_modal_exchange=nn.Linear(3,2)
    model=Toy()
    args=runtime.recipe(tmp_path)
    record=runtime.delta_record(model,args,1,metrics(),.8,{'test':True})
    saved={k:v.clone() for k,v in model.reliable_modal_exchange.state_dict().items()}
    with torch.no_grad():
        model.reliable_modal_exchange.weight.add_(10)
    runtime.restore_delta(model,args,record)
    assert torch.equal(model.reliable_modal_exchange.weight,saved['weight'])
    record['base_sha256']='changed'
    with pytest.raises(ValueError):
        runtime.restore_delta(model,args,record)


def test_replication_gate_rejects_one_seed_win_and_quality_regression():
    base={seed:metrics() for seed in (13,14,15)}
    runs={13:metrics(486),14:metrics(),15:metrics()}
    assert not report.replication_check(runs,base)['passed']
    runs={13:metrics(480),14:metrics(480),15:metrics(480)}
    assert report.replication_check(runs,base)['passed']
    runs[15]=metrics(480,weak=.62)
    assert not report.replication_check(runs,base)['passed']
