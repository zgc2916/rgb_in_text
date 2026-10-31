"""Guard complete, aligned predictions when removing DDP sampler padding."""
import importlib.util
from pathlib import Path

import pytest

_path = Path(__file__).parents[1] / 'script_eval/clip_spatial_quality_test.py'
_spec = importlib.util.spec_from_file_location('spatial_test_evaluation', _path)
evaluation = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(evaluation)
_report_spec = importlib.util.spec_from_file_location(
    'spatial_test_report', _path.with_name('clip_spatial_quality_test_report.py'))
report = importlib.util.module_from_spec(_report_spec)
_report_spec.loader.exec_module(report)


def row(index=0, **updates):
    return {'index': index, 'image': 'one.jpg', 'phrase': 'person', 'iou': .7, **updates}


def test_sampler_padding_is_removed_and_original_order_restored():
    result = evaluation.deduplicate([row(1), row(0), row(0, iou=.7000001)], 2)
    assert [x['index'] for x in result] == [0, 1]
    assert len(result) == 2


@pytest.mark.parametrize('update', [{'image': 'another.jpg'}, {'phrase': 'car'}, {'iou': .6}])
def test_inconsistent_duplicate_predictions_cannot_be_silently_overwritten(update):
    with pytest.raises(ValueError, match='disagree'):
        evaluation.deduplicate([row(), row(**update)], 1)


def test_missing_predictions_cannot_produce_an_incomplete_metric():
    with pytest.raises(ValueError, match='indices'):
        evaluation.deduplicate([row(0), row(2)], 3)


def test_paired_report_counts_rescues_and_regressions_separately():
    reference = [row(i, target_xywh=[.5, .5, .1, .1], iou=value)
                 for i, value in enumerate((.4, .8, .49))]
    candidate = [dict(x, iou=value) for x, value in zip(reference, (.6, .3, .5))]
    changes = report.paired_changes(candidate, reference)
    assert changes['gained_hits05'] == 2
    assert changes['lost_hits05'] == 1
    assert changes['net_hits05'] == 1


def test_paired_report_rejects_misaligned_expressions():
    reference = [row(target_xywh=[.5, .5, .1, .1])]
    with pytest.raises(ValueError, match='same expression'):
        report.paired_changes([dict(reference[0], phrase='other')], reference)
