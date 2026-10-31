import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from script_train.clip_structure.run_control import make_args, schedule, verify_runtime
from script_train.clip_structure.report import image_paired_bootstrap


def test_preregistered_training_budget():
    jobs = schedule()
    exploration = [x for x in jobs if x[0] == 'explore']
    formal = [x for x in jobs if x[0] == 'formal']
    assert len(exploration) == 3 and all(x[3] == 20 for x in exploration)
    assert len(formal) == 8 and all(x[3] == 120 for x in formal)
    assert {x[1] for x in formal if x[2] == 13} == {'c1', 'detail', 'boundary', 'joint'}
    assert {x[1] for x in formal if x[2] in (14, 15)} == {'c1', 'joint'}


def test_recipe_common_initialization_and_c1_warmstart(tmp_path):
    common = tmp_path / 'common.pth'
    common.touch()
    source = tmp_path / 'source.pth'
    args = argparse.Namespace(dataset='rgbtvg_flir', FusionMethod='IAFv3', modality='rgbt',
        imsize=224, target_safe_crop=True, target_scale_floor_pixels=0, retrain=str(common))
    torch.save({'args': args}, source)
    formal = make_args(source, tmp_path, 'joint', 'formal', 14)
    explore = make_args(source, tmp_path, 'joint', 'explore')
    assert formal.retrain == str(common) and not formal.clip_structure_new_only
    assert explore.retrain == str(source) and explore.clip_structure_new_only
    assert formal.enable_tir_detail and formal.enable_boundary_refiner
    assert (formal.batch_size, formal.grad_accum_steps, formal.epochs, formal.seed) == (12, 4, 120, 14)
    args.target_scale_floor_pixels = 8
    torch.save({'args': args}, source)
    with pytest.raises(ValueError, match='no scale protection'):
        make_args(source, tmp_path, 'c1', 'formal')


def test_runtime_guard(tmp_path):
    source = tmp_path / 'source.pth'
    (tmp_path / 'command.json').write_text(json.dumps({'argv': ['/different/environment/python']}))
    with pytest.raises(RuntimeError, match='recorded C1 interpreter'):
        verify_runtime(source)
    (tmp_path / 'command.json').write_text(json.dumps({'argv': [sys.executable]}))
    verify_runtime(source)


def test_image_cluster_bootstrap_preserves_expression_groups():
    reference, candidate = [], []
    for i, (image, value) in enumerate([('a', .4), ('a', .4), ('a', .4), ('b', .8)]):
        row = dict(index=i, image=image, phrase='target', target_xywh=[.5, .5, .2, .2], iou=value)
        reference.append(row)
        candidate.append({**row, 'iou': value + (.2 if image == 'a' else 0)})
    result = image_paired_bootstrap(candidate, reference, repeats=1000)
    assert result['images'] == 2 and result['expressions'] == 4
    assert result['delta_percentage_points']['acc05'] == 75
    assert result['ci95_percentage_points']['acc05'] == [0, 100]
    candidate[0]['phrase'] = 'different'
    with pytest.raises(ValueError, match='identity'):
        image_paired_bootstrap(candidate, reference)
