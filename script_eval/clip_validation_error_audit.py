"""Audit frozen validation predictions without evaluating or selecting on test."""
import argparse
import csv
import json
from pathlib import Path

import torch

from script_train.clip_lavs_control.runtime import ORIGINAL, REFERENCE, sha256, write_json
from utils.box_utils import bbox_iou, xywh2xyxy


def audit(output):
    if output.exists():
        raise FileExistsError('Refusing to overwrite an error audit')
    source = REFERENCE / 'evaluation/c1_seed13/val.json'
    recipe_path = REFERENCE / 'formal/c1_seed13/recipe.json'
    data = json.loads(source.read_text())
    args = json.loads(recipe_path.read_text())
    annotation_path = Path(args['split_root']) / args['dataset'] / f"{args['dataset']}_val.pth"
    annotations = torch.load(annotation_path, map_location='cpu', weights_only=False)
    rows = data['samples']
    if len(rows) != len(annotations):
        raise ValueError('Prediction/annotation lengths differ')
    for index, (row, annotation) in enumerate(zip(rows, annotations)):
        if row['index'] != index or Path(row['image']).name != Path(annotation[0]).name:
            raise ValueError('Prediction/annotation identity differs')
    p = torch.tensor([r['pred_xywh'] for r in rows])
    t = torch.tensor([r['target_xywh'] for r in rows])
    values = torch.tensor([r['iou'] for r in rows])
    def iou(boxes):
        return bbox_iou(xywh2xyxy(boxes).clamp(0, 1), xywh2xyxy(t))
    recomputed = iou(p)
    error = (recomputed - values).abs().max().item()
    if error > 1e-5:
        raise ValueError(f'Recomputed IoUs differ from frozen evaluation: {error}')
    centered = p.clone()
    centered[:, :2] = t[:, :2]
    resized = p.clone()
    resized[:, 2:] = t[:, 2:]
    center_oracle, size_oracle = iou(centered), iou(resized)
    small = torch.tensor([str(a[5]).lower() == 'small' for a in annotations])
    weak = torch.tensor([str(a[4]) in ('1', '2') for a in annotations])
    failed = values < .5
    short = t[:, 2:].amin(-1) * args['imsize']
    patch_area = t[:, 2:].prod(-1) * (args['imsize'] / 16) ** 2
    center_inside = ((p[:, :2] - t[:, :2]).abs() <= t[:, 2:] / 2).all(-1)
    summaries = {}
    groups = {'all': torch.ones_like(failed), 'small': small, 'normal': ~small,
              'weak': weak, 'daylight': ~weak, 'failed_acc05': failed}
    for name, mask in groups.items():
        summaries[name] = {'count': int(mask.sum()), 'acc05': float((values[mask] >= .5).double().mean()),
            'acc07': float((values[mask] >= .7).double().mean()), 'miou': float(values[mask].double().mean()),
            'failures_acc05': int((mask & failed).sum()), 'short_side_median_at224': float(short[mask].median()),
            'short_side_below_one_patch': int((short[mask] < 16).sum()),
            'area_below_one_patch': int((patch_area[mask] < 1).sum()),
            'predicted_center_inside_gt': int((mask & center_inside).sum()),
            'center_oracle_rescued_failures': int((mask & failed & (center_oracle >= .5)).sum()),
            'size_oracle_rescued_failures': int((mask & failed & (size_oracle >= .5)).sum())}
    coverage = {}
    for scale, minimum in ((2., 0), (2., 32), (3., 32), (3., 64)):
        size = (p[:, 2:] * scale).clamp(min=minimum / args['imsize'], max=1.)
        center = torch.maximum(size / 2, torch.minimum(p[:, :2], 1 - size / 2))
        crop_lo, crop_hi = center - size / 2, center + size / 2
        full = ((crop_lo <= t[:, :2] - t[:, 2:] / 2) & (crop_hi >= t[:, :2] + t[:, 2:] / 2)).all(-1)
        contains_center = ((crop_lo <= t[:, :2]) & (crop_hi >= t[:, :2])).all(-1)
        coverage[f'scale{scale:g}_floor{minimum}'] = {'prediction_only_crop_policy': True,
            'coordinate_space': 'normalized padded 224 canvas; no images cropped or re-encoded',
            'gt_fully_contained_all': int(full.sum()), 'gt_fully_contained_failed': int((full & failed).sum()),
            'gt_center_contained_failed': int((contains_center & failed).sum()),
            'mean_normalized_crop_area': float(size.prod(-1).mean())}
    output.mkdir(parents=True)
    table = []
    for index, (row, ann) in enumerate(zip(rows, annotations)):
        table.append({'index': index, 'image': row['image'], 'phrase': row['phrase'],
            'size_label': ann[5], 'weak': bool(weak[index]), 'iou': row['iou'],
            'failed_acc05': bool(failed[index]), 'gt_short_side_224': float(short[index]),
            'gt_area_in_patch_units': float(patch_area[index]),
            'center_oracle_iou': float(center_oracle[index]), 'size_oracle_iou': float(size_oracle[index]),
            'center_inside_gt': bool(center_inside[index])})
    with (output / 'samples.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=table[0].keys())
        writer.writeheader()
        writer.writerows(table)
    result = {'purpose': 'validation-only error decomposition before designing a new method',
        'test_used': False, 'new_model_evaluated': False, 'oracle_is_not_an_achievable_model_result': True,
        'sources': {str(p): sha256(p) for p in (source, recipe_path, annotation_path, Path(__file__))},
        'recomputed_iou_max_error': error, 'groups': summaries, 'crop_coverage': coverage,
        'small_fraction_of_acc05_failures': int((small & failed).sum()) / int(failed.sum()),
        'interpretation': ['Small-target center localization is the dominant observed error pattern.',
            'The diagnostic cannot establish that sensor misalignment caused the errors.',
            'A short side below one patch does not imply the entire object occupies less than one patch.',
            'Crop coverage only measures geometric feasibility, not expected accuracy gains.']}
    write_json(output / 'report.json', result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ORIGINAL / 'output_clip_analysis/2026-10-27_c1_validation_errors')
    audit(parser.parse_args().output.resolve())
