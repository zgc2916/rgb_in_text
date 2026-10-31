import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from script_train.clip_soft_control import control, runtime


def metrics(accu, accu_07=.59, miou=.67):
    return {'accu': accu, 'accu_07': accu_07, 'miou': miou}


def test_three_modes_share_formal_recipe_and_keep_same_public_initializer(tmp_path):
    recipes = [vars(runtime.recipe(tmp_path, v)) for v in runtime.VARIANTS]
    assert len(recipes) == 3
    assert all({k for k in recipes[0] if recipes[0][k] != a[k]} == {'visual_exchange'} for a in recipes[1:])
    a = recipes[0]
    assert a['batch_size'] * a['grad_accum_steps'] * a['world_size'] == 96
    assert a['epochs'] == 120 and a['FusionMethod'] == 'IAFv3'
    assert 'merged_fixed_best_checkpoint_peft0111.pth' in a['retrain']
    assert not a['enable_tir_detail'] and not a['enable_boundary_refiner']


def test_promotion_requires_baseline_gains_and_reports_mechanism_independently():
    base = metrics(476/608)
    runs = {'global_gate': metrics(479/608), 'text_global_gate': metrics(480/608),
            'text_soft_spatial': metrics(482/608, .60, .68)}
    result = control.choose_promotion(base, runs)
    assert result['selected_variant'] == 'text_soft_spatial' and not result['test_used']
    assert result['text_dominates_visual'] and result['spatial_dominates_text_global']
    runs['text_soft_spatial'] = metrics(485/608, .58, .68)
    assert control.choose_promotion(base, runs)['selected_variant'] == 'text_global_gate'


def test_gate_rejects_both_small_gain_and_quality_tradeoff():
    base = metrics(476/608)
    runs = {'global_gate': metrics(477/608), 'text_global_gate': metrics(479/608),
            'text_soft_spatial': metrics(486/608, .58)}
    assert control.choose_promotion(base, runs)['selected_variant'] is None


def test_checkpoint_choice_is_first_strict_maximum(tmp_path):
    rows = [{'epoch': i, 'val_accu': .8 if i in (10, 70) else .7} for i in range(120)]
    (tmp_path/'log.txt').write_text(''.join(json.dumps(x)+'\n' for x in rows))
    runtime.write_json(tmp_path/'best.json', {'epoch': 10, 'accu': .8})
    assert control.first_best(tmp_path)['epoch'] == 10
    runtime.write_json(tmp_path/'best.json', {'epoch': 70, 'accu': .8})
    with pytest.raises(ValueError): control.first_best(tmp_path)
