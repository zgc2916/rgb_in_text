import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
from script_train.clip_lavs_control import control, runtime


def stats(accu, accu_07=.59, miou=.67):
    return {'accu': accu, 'accu_07': accu_07, 'miou': miou}


def test_promotion_rejects_accuracy_tradeoff_and_small_gain():
    baseline = stats(476/608)
    runs = {'local': stats(479/608), 'text_local': stats(483/608, .58)}
    result = control.choose_promotion(baseline, runs)
    assert result['selected_variant'] is None and not result['test_used']


def test_promotion_reports_text_mechanism_separately():
    baseline = stats(476/608)
    runs = {'local': stats(481/608), 'text_local': stats(484/608, .60, .68)}
    result = control.choose_promotion(baseline, runs)
    assert result['selected_variant'] == 'text_local'
    assert result['text_gate_dominates_local']
    result = control.choose_promotion(baseline, {'local': runs['text_local'], 'text_local': runs['local']})
    assert result['selected_variant'] == 'local' and not result['text_gate_dominates_local']


def test_first_best_tie_uses_first_epoch_and_requires_complete_run(tmp_path):
    rows = [{'epoch': i, 'val_accu': .8 if i in (20, 60) else .7} for i in range(120)]
    (tmp_path/'log.txt').write_text(''.join(json.dumps(x)+'\n' for x in rows))
    runtime.write_json(tmp_path/'best.json', {'epoch': 20, 'accu': .8})
    assert control.first_best(tmp_path)['epoch'] == 20
    runtime.write_json(tmp_path/'best.json', {'epoch': 60, 'accu': .8})
    with pytest.raises(ValueError):
        control.first_best(tmp_path)


@pytest.mark.parametrize('crash_point', ['after_first_rename', 'after_both_renames', 'after_new_best'])
def test_best_journal_recovers_only_committed_best(tmp_path, crash_point):
    (tmp_path/'previous_best_checkpoint.pth').write_bytes(b'committed')
    old = {'epoch': 2, 'accu': .7}
    if crash_point == 'after_first_rename':
        runtime.write_json(tmp_path/'best.json', old)
    else:
        runtime.write_json(tmp_path/'previous_best.json', old)
    if crash_point == 'after_new_best':
        (tmp_path/'best_checkpoint.pth').write_bytes(b'uncommitted')
        runtime.write_json(tmp_path/'best.json', {'epoch': 4, 'accu': .8})
    runtime.recover_best(tmp_path, 3)
    assert (tmp_path/'best_checkpoint.pth').read_bytes() == b'committed'
    assert json.loads((tmp_path/'best.json').read_text()) == old


def test_trim_log_keeps_committed_epochs_and_preserves_interrupted_log(tmp_path):
    rows = [{'epoch': i} for i in range(5)]
    (tmp_path/'log.txt').write_text(''.join(json.dumps(x)+'\n' for x in rows))
    runtime.trim_log(tmp_path, 2)
    assert [json.loads(x)['epoch'] for x in (tmp_path/'log.txt').read_text().splitlines()] == [0, 1, 2]
    assert list(tmp_path.glob('log.interrupted.*.txt'))


def test_matched_recipes_preserve_reference_training_and_differ_only_in_mode(tmp_path):
    local = vars(runtime.recipe(tmp_path, 'local'))
    text = vars(runtime.recipe(tmp_path, 'text_local'))
    assert {k for k in local if local[k] != text[k]} == {'visual_exchange'}
    assert local['batch_size'] * local['world_size'] * local['grad_accum_steps'] == 96
    assert local['epochs'] == 120 and local['FusionMethod'] == 'IAFv3'
    assert not local['enable_tir_detail'] and not local['enable_boundary_refiner']
