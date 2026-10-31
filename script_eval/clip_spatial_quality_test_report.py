"""Paired held-out analysis of the validation-selected spatial-quality suite."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np


def metrics(rows):
    if not rows:
        return {'total': 0}
    values = np.asarray([row['iou'] for row in rows], dtype=np.float64)
    return {'total': len(rows), 'images': len({row['image'] for row in rows}),
            'correct_05': int((values >= .5).sum()), 'correct_07': int((values >= .7).sum()),
            'acc_05': float((values >= .5).mean()), 'acc_07': float((values >= .7).mean()),
            'miou': float(values.mean())}


def paired_changes(candidate, reference):
    if len(candidate) != len(reference) or not candidate:
        raise ValueError('Paired analysis requires complete equal-length prediction sets')
    for a, b in zip(candidate, reference):
        if any(a[key] != b[key] for key in ('index', 'image', 'phrase', 'target_xywh')):
            raise ValueError('Paired predictions do not refer to the same expression')
    a = np.asarray([row['iou'] for row in candidate])
    b = np.asarray([row['iou'] for row in reference])
    return {'gained_hits05': int(((a >= .5) & (b < .5)).sum()),
            'lost_hits05': int(((a < .5) & (b >= .5)).sum()),
            'net_hits05': int((a >= .5).sum() - (b >= .5).sum()),
            'gained_hits07': int(((a >= .7) & (b < .7)).sum()),
            'lost_hits07': int(((a < .7) & (b >= .7)).sum()),
            'iou_improved_gt001': int((a - b > .01).sum()),
            'iou_degraded_gt001': int((a - b < -.01).sum()),
            'mean_iou_delta_percentage_points': float((a - b).mean() * 100)}


def groups(rows):
    return {'all': rows,
            'small': [row for row in rows if row['size_label'] == 'small'],
            'normal': [row for row in rows if row['size_label'] != 'small'],
            'weak': [row for row in rows if row['weak']],
            'daylight': [row for row in rows if not row['weak']]}


def local_diagnostics(rows):
    if 'proposal_ious' not in rows[0]:
        return None
    final = np.asarray([row['iou'] for row in rows])
    coarse = np.asarray([row['coarse_iou'] for row in rows])
    proposal = np.asarray([row['proposal_ious'] for row in rows])
    weight = np.asarray([row['local_weights'] for row in rows])
    predicted = np.asarray([row['predicted_gain'] for row in rows])
    gain = proposal - coarse[:, None]
    positive = predicted > 0
    return {'mean_rgb_weight': float(weight[:, 0].mean()), 'mean_tir_weight': float(weight[:, 1].mean()),
            'mean_global_weight': float((1 - weight.sum(-1)).mean()),
            'both_local_weights_zero_count': int((weight.sum(-1) == 0).sum()),
            'predicted_positive_view_count': int(positive.sum()),
            'positive_prediction_actual_gain_precision': float((gain[positive] > 0).mean()) if positive.any() else None,
            'proposal_improves_gt001_view_count': int((gain > .01).sum()),
            'local_best_improves_gt001_sample_count': int((gain.max(-1) > .01).sum()),
            'mean_iou_delta_vs_same_model_coarse_percentage_points': float((final - coarse).mean() * 100),
            'gained_hits05_vs_same_model_coarse': int(((final >= .5) & (coarse < .5)).sum()),
            'lost_hits05_vs_same_model_coarse': int(((final < .5) & (coarse >= .5)).sum())}


def build(output):
    plan = json.loads((output / 'evaluation_plan.json').read_text())
    root = Path(plan['training_root'])
    sys.path.insert(0, str(root / 'source'))
    from script_train.clip_structure.report import image_paired_bootstrap
    from script_train.clip_spatial_quality_control.runtime import sha256, write_json
    if json.loads((output / 'status.json').read_text())['phase'] != 'complete':
        raise RuntimeError('Full evaluation has not completed')
    results, evaluations = {}, {}
    for name in plan['checkpoints']:
        evaluations[name] = {split: json.loads((output / name / f'{split}.json').read_text())
                             for split in ('val', 'test')}
        results[name] = {split: {label: metrics(rows) for label, rows in groups(data['samples']).items()}
                         for split, data in evaluations[name].items()}
    report = {'checkpoint_selection': plan['selection'], 'evaluation_plan_sha256': sha256(output / 'evaluation_plan.json'),
              'test_used_for_selection': False, 'training_seed': 13, 'multi_seed_replication': False,
              'runs': results, 'paired': {}, 'local_diagnostics': {}, 'original_c1_failure_profile': {},
              'uncertainty': 'Image-cluster paired percentile bootstrap, 5000 repeats; fixed trained models, no seed uncertainty',
              'test_evaluation_does_not_change_validation_selection': True}
    for name in evaluations:
        rows = evaluations[name]['test']['samples']
        report['local_diagnostics'][name] = local_diagnostics(rows)
        if name == 'original_c1':
            failed = [row for row in rows if row['iou'] < .5]
            report['original_c1_failure_profile'] = {
                'failed_hits05': len(failed), 'small_failures': sum(row['size_label'] == 'small' for row in failed),
                'small_fraction_of_failures': sum(row['size_label'] == 'small' for row in failed) / len(failed),
                'short_side_below16_at224_failures': sum(min(row['target_xywh'][2:]) * 224 < 16 for row in failed)}
            continue
        report['paired'][name] = {}
        for reference in ('original_c1', 'c1_continue'):
            if reference == name:
                continue
            reference_rows = evaluations[reference]['test']['samples']
            paired = {'all': {**paired_changes(rows, reference_rows),
                       'bootstrap': image_paired_bootstrap(rows, reference_rows)}}
            for label in ('small', 'normal', 'weak', 'daylight'):
                a, b = groups(rows)[label], groups(reference_rows)[label]
                paired[label] = paired_changes(a, b)
            report['paired'][name][reference] = paired
    prior_path = root.parents[1] / 'output_clip_structure_control/2026-10-05_c1_detail_boundary/evaluation/c1_seed13/test.json'
    if prior_path.exists():
        prior = json.loads(prior_path.read_text())['samples']
        fresh = evaluations['original_c1']['test']['samples']
        report['historical_c1_reproduction'] = paired_changes(fresh, prior)
        report['historical_c1_reproduction']['max_abs_iou_error'] = max(abs(a['iou'] - b['iou']) for a, b in zip(fresh, prior))
    write_json(output / 'analysis.json', report)
    table = []
    original = evaluations['original_c1']['test']['samples']
    for i, row in enumerate(original):
        record = {key: row[key] for key in ('index', 'image', 'phrase', 'size_label', 'weak')}
        for name, splits in evaluations.items():
            record[name + '_iou'] = splits['test']['samples'][i]['iou']
        table.append(record)
    with (output / 'paired_samples.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    print(json.dumps({'test': {k: v['test']['all'] for k, v in results.items()},
                      'local_diagnostics': report['local_diagnostics']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('output_clip_analysis/2026-10-28_spatial_quality_test'))
    build(parser.parse_args().output.resolve())
