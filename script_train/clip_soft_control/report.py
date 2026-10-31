"""Frozen phase1 validation and matched text/spatial mechanism comparisons."""
import json
from pathlib import Path

from .runtime import REFERENCE, sha256
from script_train.clip_structure.report import metrics, image_paired_bootstrap


def report(root):
    root = Path(root)
    selected = json.loads((root/'selected_models.json').read_text())
    result = {'selection': 'first strict maximum validation Acc@0.5',
              'promotion': json.loads((root/'promotion.json').read_text()),
              'replaced': ['external RGB/TIR exchange'],
              'preserved': ['CLIP internal language bridges', 'IAFv3', 'six VL encoder layers'],
              'novelty_status': 'task-specific hypothesis, not a demonstrated contribution',
              'test_used': False, 'test_previously_observed': True,
              'phase1_only': True, 'runs': {}, 'mechanism_comparisons': {}}
    baseline = json.loads((REFERENCE/'evaluation/c1_seed13/val.json').read_text())['samples']
    predictions = {}
    for item in selected:
        if item['seed'] != 13 or sha256(item['checkpoint']) != item['sha256']:
            raise ValueError('Frozen phase1 checkpoint or seed differs')
        evaluation = json.loads((root/'evaluation'/item['name']/'val.json').read_text())
        if evaluation['source']['epoch'] != item['epoch']:
            raise ValueError('Evaluation used another checkpoint epoch')
        rows = evaluation['samples']
        predictions[item['variant']] = rows
        result['runs'][item['name']] = {'epoch': item['epoch'], 'sha256': item['sha256'],
            'validation': metrics(rows), 'validation_groups': item['validation'],
            'paired_c1': image_paired_bootstrap(rows, baseline)}
    for label, candidate, control in (
            ('text_vs_visual', 'text_global_gate', 'global_gate'),
            ('spatial_vs_text_global', 'text_soft_spatial', 'text_global_gate'),
            ('full_vs_visual', 'text_soft_spatial', 'global_gate')):
        result['mechanism_comparisons'][label] = image_paired_bootstrap(predictions[candidate], predictions[control])
    return result
