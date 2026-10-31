"""Guard expensive geometry comparisons against mismatched controls and peaks."""

import argparse
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from script_train.infmae_experiments.run_clip_geometry_control import (
    assess_quality, recipe_differences, saved_best, validate_reference,
)


def _row(epoch, acc05=0.77, acc07=0.56, weak07=0.46):
    return {'epoch': epoch, 'accu': acc05, 'accu_07': acc07, 'miou': 0.65,
            'accu_07_weak': weak07, 'accu_07_small': 0.34}


def _reference(tmp_path):
    directory = tmp_path / 'control' / 'batch12'
    directory.mkdir(parents=True)
    source = tmp_path / 'recipe_source.pth'
    args = argparse.Namespace(
        epochs=120, seed=13, lr=1e-4, batch_size=12, grad_accum_steps=4,
        world_size=2, FusionMethod='IAFv3', match_infmae_lora_schedule=True,
        retrain='/common/mmvg.pth', vl_backbone='clip', contrastive_loss='clip',
        infmae_alignment_mode='none', modality='rgbt', aug_crop=True, aug_scale=True,
        output_dir=str(directory), target_safe_crop=False, target_scale_floor_pixels=0.0,
    )
    config = {'epochs': 120, 'seed': 13, 'lr': 1e-4, 'batch_size_per_gpu': 12,
              'grad_accum_steps': 4, 'world_size': 2, 'nominal_effective_batch': 96,
              'fusion_method': 'IAFv3', 'match_infmae_lora_schedule': True,
              'initialization': args.retrain, 'reference_recipe': str(source)}
    (directory / 'control_configuration.json').write_text(json.dumps(config))
    recorded = vars(args).copy()
    recorded.pop('target_safe_crop')
    recorded.pop('target_scale_floor_pixels')
    (directory.parent / 'recipe.json').write_text(json.dumps({'args': recorded}))
    (directory / 'best_checkpoint.pth').touch()
    rows = [{f'validation_{key}' if key != 'epoch' else key: value
             for key, value in _row(epoch).items()} for epoch in range(120)]
    (directory / 'log.txt').write_text('\n'.join(json.dumps(row) for row in rows) + '\n')
    cli = argparse.Namespace(source=source, control_dir=directory, epochs=120, seed=13)
    return cli, args


def test_default_geometry_matches_legacy_recipe_and_lr_changes_are_detected():
    legacy = {'lr': 1e-4, 'output_dir': 'old'}
    candidate = {**legacy, 'output_dir': 'new', 'target_safe_crop': False,
                 'target_scale_floor_pixels': 0.0}
    assert recipe_differences(legacy, candidate) == {}
    candidate['lr'] = 5e-6
    assert 'lr' in recipe_differences(legacy, candidate)


def test_existing_complete_control_is_accepted_but_changed_training_budget_is_rejected(tmp_path):
    cli, args = _reference(tmp_path)
    assert validate_reference(cli, args)['epochs_completed'] == 120
    args.grad_accum_steps = 8
    with pytest.raises(ValueError, match='recipe has changed'):
        validate_reference(cli, args)


def test_duplicate_epochs_cannot_serve_as_a_completed_control(tmp_path):
    cli, args = _reference(tmp_path)
    path = cli.control_dir / 'log.txt'
    rows = path.read_text().splitlines()
    rows[-1] = rows[-2]
    path.write_text('\n'.join(rows) + '\n')
    with pytest.raises(ValueError, match='complete original control'):
        validate_reference(cli, args)


def test_tied_acc05_keeps_the_actual_first_saved_epoch():
    rows = [_row(111, acc05=0.78), _row(119, acc05=0.78, acc07=0.60)]
    assert saved_best(rows)['epoch'] == 111


def test_overall_gain_does_not_hide_low_light_high_iou_regression():
    reference = _row(111)
    regressed = _row(110, acc05=0.79, acc07=0.58, weak07=0.44)
    assert not assess_quality(reference, regressed)['eligible_for_seed_replication']
    improved = _row(110, acc05=0.79, acc07=0.58, weak07=0.48)
    assert assess_quality(reference, improved)['eligible_for_seed_replication']
