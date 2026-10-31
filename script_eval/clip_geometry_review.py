"""Verify cached fixed-model predictions and review the CLIP crop comparison."""

import argparse
import json
from pathlib import Path

import torch

from script_train.infmae_experiments.run_clip_geometry_control import (
    assess_quality, saved_best, sha256_file, validation_rows,
)
from script_train.infmae_experiments.run_scale_ablation import timestamp, write_json
from script_eval.infmae_ddp_probe import summarize
from utils.box_utils import bbox_iou, xywh2xyxy


def metric_deltas(candidate, reference):
    return {key: (candidate[key] - reference[key]) * 100
            for key in ('acc_05', 'acc_07', 'miou')}


def paired(candidate, reference):
    result = {}
    for threshold in (0.5, 0.7):
        improved = sum(a['iou'] >= threshold and b['iou'] < threshold
                       for a, b in zip(candidate, reference))
        regressed = sum(a['iou'] < threshold and b['iou'] >= threshold
                       for a, b in zip(candidate, reference))
        result[str(threshold)] = {'candidate_correct_reference_wrong': improved,
                                  'candidate_wrong_reference_correct': regressed,
                                  'net_correct': improved - regressed}
    return result


def verify_cache(report, annotations):
    rows = report['samples']
    if [row['index'] for row in rows] != list(range(len(annotations))):
        raise AssertionError('Predictions do not cover the ordered annotation split exactly once')
    for row, annotation in zip(rows, annotations):
        if row['image'] != str(annotation[0]) or row['phrase'] != annotation[3].lower():
            raise AssertionError('Prediction/annotation image or expression mismatch')
    pred = torch.tensor([row['pred_xywh'] for row in rows], dtype=torch.float32)
    target = torch.tensor([row['target_xywh'] for row in rows], dtype=torch.float32)
    values = bbox_iou(xywh2xyxy(pred).clamp(0, 1), xywh2xyxy(target))
    saved = torch.tensor([row['iou'] for row in rows], dtype=torch.float32)
    difference = float((values - saved).abs().max())
    if difference != 0 or not torch.isfinite(values).all():
        raise AssertionError('Original CPU metric formula does not reproduce cached GPU IoUs')
    stats = summarize(rows)
    if stats != report['summary']:
        raise AssertionError('Cached metric summary does not match its individual predictions')
    return {'ordered_coverage_exact': True, 'max_absolute_iou_difference': difference,
            'threshold_counts_exact': True}


def verify_pair(candidate, reference):
    if len(candidate) != len(reference):
        raise AssertionError('Compared models cover different split sizes')
    for a, b in zip(candidate, reference):
        if any(a[key] != b[key] for key in ('index', 'image', 'phrase', 'target_xywh')):
            raise AssertionError('Compared indices, images, expressions or GT boxes differ')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--training-root', type=Path, required=True)
    cli = parser.parse_args()
    root, training = cli.output_root.resolve(), cli.training_root.resolve()
    selected = json.loads((root / 'selected_model.json').read_text())
    state = json.loads((root / 'status.json').read_text())
    if state.get('returncode') != 0 or state.get('test_forward_count') != 1:
        raise ValueError('Require one successfully completed fixed-model test evaluation')
    frozen_hash = sha256_file(selected['frozen_checkpoint'])
    if frozen_hash != selected['frozen_checkpoint_sha256']:
        raise AssertionError('Frozen checkpoint changed during evaluation')
    recipes = json.loads((training / 'recipe.json').read_text())
    annotation_root = Path(recipes['variants']['safe_crop']['split_root']) / 'rgbtvg_flir'
    repository = Path(__file__).resolve().parents[1]
    references = {
        'original_batch12': repository / 'output_clip_control_test/2026-10-04_seed13_b12_epoch111',
        'batch6': repository / 'output_clip_control_test/2026-10-04_seed13_epoch95',
        'infmae': repository / 'output_infmae_test_review/2026-10-03_final_selected',
    }
    paper = json.loads((references['original_batch12'] / 'summary.json').read_text())['paper_reference']
    summary = {'created_at': timestamp(), 'model_identity': selected['model_identity'],
               'selected_epoch': selected['selected_epoch'], 'variant': 'safe_crop',
               'selection_record': str(root / 'selected_model.json'),
               'frozen_checkpoint_sha256_verified_after_evaluation': frozen_hash,
               'test_used_to_select_this_checkpoint': False,
               'test_forward_evaluations_this_round': 1, 'cross_seed_verified': False,
               'paper_reference': paper, 'protocol': {}, 'annotation_files': {},
               'cached_cpu_metric_verification': {}, 'references': {}, 'paired': {},
               'groups': {}, 'group_definitions': {
                   'weak_light_codes_1_2': 'Full split lighting codes1/2, not the separate TestB file',
                   'small_label': 'Published small size label', 'groups_overlap': True}}
    for split, expected in (('val', 608), ('test', 2104)):
        report = json.loads((root / f'{split}.json').read_text())
        annotation_path = annotation_root / f'rgbtvg_flir_{split}.pth'
        annotations = torch.load(annotation_path, map_location='cpu', weights_only=False)
        if len(annotations) != expected or report['source']['epoch'] != selected['selected_epoch']:
            raise AssertionError('Unexpected split size or evaluated checkpoint')
        if report['protocol']['duplicate_padding_removed'] != 0:
            raise AssertionError('This split should have no padded DDP repeats')
        summary[split] = report['summary']
        summary['protocol'][split] = report['protocol']
        summary['annotation_files'][split] = {'path': str(annotation_path), 'count': len(annotations),
                                               'sha256': sha256_file(annotation_path)}
        summary['cached_cpu_metric_verification'][split] = {
            'safe_crop': verify_cache(report, annotations)}
        models = {'safe_crop': report['samples']}
        summary['references'][split], summary['paired'][split] = {}, {}
        for name, directory in references.items():
            cached = json.loads((directory / f'{split}.json').read_text())
            summary['cached_cpu_metric_verification'][split][name] = verify_cache(cached, annotations)
            verify_pair(report['samples'], cached['samples'])
            models[name] = cached['samples']
            summary['references'][split][name] = {
                'epoch': cached['source']['epoch'], 'metrics': cached['summary'],
                'same_samples_and_targets_verified': True,
                'candidate_minus_reference_percentage_points': metric_deltas(report['summary'], cached['summary'])}
            summary['paired'][split][name] = paired(report['samples'], cached['samples'])
        masks = {
            'weak_light_codes_1_2': [str(row[4]) in {'1', '2'} for row in annotations],
            'other_light': [str(row[4]) not in {'1', '2'} for row in annotations],
            'small_label': [str(row[5]).lower() == 'small' for row in annotations],
            'normal_label': [str(row[5]).lower() != 'small' for row in annotations],
        }
        summary['groups'][split] = {}
        for group, mask in masks.items():
            group_rows = {name: [row for row, include in zip(rows, mask) if include]
                          for name, rows in models.items()}
            summary['groups'][split][group] = {name: summarize(rows) for name, rows in group_rows.items()}
            summary['groups'][split][group]['paired_vs_original_batch12'] = paired(
                group_rows['safe_crop'], group_rows['original_batch12'])
    recorded = selected['recorded_validation']
    measured = summary['val']
    if measured['correct_05'] != round(recorded['accu'] * 608) or (
            measured['correct_07'] != round(recorded['accu_07'] * 608)):
        raise AssertionError('Validation threshold counts did not reproduce the frozen model')
    difference = abs(measured['miou'] - recorded['miou'])
    if difference > 1e-6:
        raise AssertionError('Validation mean IoU did not reproduce the frozen model')
    summary['validation_reproduced'] = True
    summary['validation_miou_absolute_difference'] = difference
    summary['safe_crop_minus_paper_test_percentage_points'] = {
        'acc_05': summary['test']['acc_05'] * 100 - paper['test_acc05_percent'],
        'acc_07': summary['test']['acc_07'] * 100 - paper['test_acc07_percent']}
    summary['validation_minus_test_percentage_points'] = metric_deltas(summary['val'], summary['test'])
    summary['limitations'] = [
        'Single seed13; no claim of stable cross-seed improvement.',
        'Local matched recipe; not an exact author-weight or complete paper-recipe reproduction.',
        'Multiple expressions share images; paired counts are descriptive.',
        'The scale-protected variant is still training and is not selected using this test result.',
    ]
    original_rows = validation_rows(references['original_batch12'].parent.parent /
                                    'output_clip_batch_control/2026-10-04_seed13_b12_a4/batch12')
    baseline = saved_best(original_rows)
    training_review = {'checked_at': timestamp(), 'selection': 'first_strict_acc05_maximum',
                       'original_saved_best': baseline, 'variants': {}, 'test_used_for_selection': False}
    for name in ('safe_crop', 'safe_crop_floor8'):
        rows = validation_rows(training / name)
        best = saved_best(rows)
        training_review['variants'][name] = {'epochs_completed': len(rows), 'saved_best': best,
                                             'latest_epoch': rows[-1] if rows else None,
                                             'quality_against_original': assess_quality(baseline, best)}
    write_json(root / 'training_review.json', training_review)
    write_json(root / 'summary.json', summary)
    state.update(phase='complete', review_completed_at=timestamp(),
                 validation_reproduced=True, frozen_checkpoint_unchanged=True)
    write_json(root / 'status.json', state)
    write_json(training / 'safe_crop_test_evaluation.json', {
        'recorded_at': timestamp(), 'selected_epoch': selected['selected_epoch'],
        'summary': str(root / 'summary.json'), 'test': summary['test'],
        'test_used_for_training_or_selection': False})
    print(json.dumps({'val': summary['val'], 'test': summary['test'],
                      'vs_original_batch12': summary['references']['test']['original_batch12'],
                      'vs_paper': summary['safe_crop_minus_paper_test_percentage_points'],
                      'groups': summary['groups']['test']}, indent=2))


if __name__ == '__main__':
    main()
