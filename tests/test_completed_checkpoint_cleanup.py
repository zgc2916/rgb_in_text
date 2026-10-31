import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1]))
from script_train.clip_training_control import cleanup_completed as c


def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(c, 'BASE', tmp_path)
    root = tmp_path / 'completed'
    root.mkdir()
    c.write_json(root / 'status.json', {'phase': 'gate_not_passed', 'current_job': None})
    monkeypatch.setattr(c, 'SUITES', (root,))
    monkeypatch.setattr(c, 'TEST_COPY', root)
    path = root / 'explore/case/checkpoint.pth'
    kept = root / 'frozen/reference.pth'
    c.atomic_checkpoint(kept, {'model': {'weight': torch.ones(2)}})
    state = {'model_delta': {'weight': torch.ones(2)}, 'epoch': 19,
             'args': SimpleNamespace(seed=13), 'val_stats': {'unused_mask': float('nan')},
             'optimizer': {'state': torch.zeros(100)}, 'lr_scheduler': {}, 'rng_by_rank': []}
    c.atomic_checkpoint(path, state)
    plan = {'audited_at': 1, 'protected': {str(kept): c.sha256(kept)},
            'files': [{**c.info(path), 'root': str(root), 'kind': 'compact_completed_last',
                       'destination': str(root / 'compact_last/case_epoch19.pth'), 'deleted': False}]}
    manifest = tmp_path / 'manifest.json'
    c.write_json(manifest, plan)
    return root, path, kept, manifest, state


def test_compaction_round_trips_all_inference_fields_before_removing_old_state(tmp_path, monkeypatch):
    root, path, kept, manifest, state = setup(tmp_path, monkeypatch)
    digest = c.sha256(kept)
    c.execute(manifest)
    assert not path.exists() and c.sha256(kept) == digest
    restored = torch.load(root / 'compact_last/case_epoch19.pth', weights_only=False)
    assert c.same_value(restored, {k: v for k, v in state.items() if k not in c.RESUME_FIELDS})
    plan = json.loads(manifest.read_text())
    assert plan['finished'] and plan['protected_checksums_verified'] and plan['deleted_files'] == 1


def test_modified_candidate_is_not_deleted(tmp_path, monkeypatch):
    root, path, kept, manifest, state = setup(tmp_path, monkeypatch)
    state['model_delta']['weight'].add_(1.)
    c.atomic_checkpoint(path, state)
    with pytest.raises(ValueError, match='changed since audit'):
        c.execute(manifest)
    assert path.exists() and kept.exists() and not (root / 'compact_last').exists()


def test_active_experiment_is_not_deleted(tmp_path, monkeypatch):
    root, path, kept, manifest, _ = setup(tmp_path, monkeypatch)
    c.write_json(root / 'status.json', {'phase': 'training_controls', 'current_job': 'case'})
    with pytest.raises(RuntimeError, match='inactive'):
        c.execute(manifest)
    assert path.exists() and kept.exists()


def test_symlink_candidate_is_rejected(tmp_path, monkeypatch):
    root, path, kept, manifest, _ = setup(tmp_path, monkeypatch)
    link = root / 'linked.pth'
    link.symlink_to(kept)
    with pytest.raises(ValueError, match='escapes'):
        c.info(link)


def test_learned_anchor_output_cannot_be_deleted_as_initial_c1():
    prefix = 'region_anchor_exchange.'
    base = {'model': {'cross_fusion_layers_vt.weight': torch.ones(2)}}
    delta = {'cross_fusion_layers_vt.weight': torch.ones(2)}
    for branch in ('rgb_output', 'tir_output'):
        for field in ('weight', 'bias'):
            delta[prefix + branch + '.' + field] = torch.zeros(2)
    checkpoint = {'epoch': -1, 'model_delta': delta,
                  'args': SimpleNamespace(visual_exchange='anchor_text_separate')}
    c.verify_initial_c1(checkpoint, base)
    delta[prefix + 'rgb_output.weight'][0] = 1.
    with pytest.raises(ValueError, match='equivalent'):
        c.verify_initial_c1(checkpoint, base)
