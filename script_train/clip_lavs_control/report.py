"""Same-checkpoint metrics and paired image bootstrap for external exchange."""
import json
from pathlib import Path

import numpy as np

from .runtime import REFERENCE, VARIANTS, sha256
from script_train.clip_structure.report import metrics, image_paired_bootstrap, bootstrap_image_deltas


def report(root):
    root = Path(root)
    selected = json.loads((root/'selected_models.json').read_text())
    result = {'selection': 'first strict maximum validation Acc@0.5',
              'promotion': json.loads((root/'promotion.json').read_text()),
              'replaced': ['external RGB/TIR exchange'],
              'preserved': ['CLIP internal language bridges', 'IAFv3', 'six VL encoder layers'],
              'novelty_status': 'local attention and gating are prior techniques; task-specific contribution requires evidence',
              'test_previously_observed': True, 'runs': {}, 'three_seed': {}}
    predictions = {}
    for item in selected:
        name, seed = item['name'], item['seed']
        if sha256(item['checkpoint']) != item['sha256']:
            raise ValueError('Frozen checkpoint changed')
        record = {'epoch': item['epoch'], 'sha256': item['sha256'], 'splits': {}}
        for split in ('val', 'test'):
            path = root/'evaluation'/name/f'{split}.json'
            if not path.exists():
                continue
            evaluation = json.loads(path.read_text())
            if evaluation['source']['epoch'] != item['epoch']:
                raise ValueError('Evaluation used another checkpoint epoch')
            rows = evaluation['samples']
            baseline = json.loads((REFERENCE/'evaluation'/f'c1_seed{seed}'/f'{split}.json').read_text())['samples']
            predictions[item['variant'], seed, split] = rows
            record['splits'][split] = {**metrics(rows), 'paired_c1': image_paired_bootstrap(rows, baseline)}
        result['runs'][name] = record
    for split in ('val', 'test'):
        for variant in VARIANTS:
            if not all((variant, seed, split) in predictions for seed in (13, 14, 15)):
                continue
            rows_by_seed = [predictions[variant, seed, split] for seed in (13, 14, 15)]
            baseline_by_seed = [json.loads((REFERENCE/'evaluation'/f'c1_seed{seed}'/f'{split}.json').read_text())['samples']
                                for seed in (13, 14, 15)]
            def difference(a, b):
                image_paired_bootstrap(a, b, repeats=1)  # validates annotation identity
                return [[float(x['iou'] >= .5) - float(y['iou'] >= .5),
                         float(x['iou'] >= .7) - float(y['iou'] >= .7), x['iou'] - y['iou']] for x, y in zip(a, b)]
            delta = np.asarray([difference(a, b) for a, b in zip(rows_by_seed, baseline_by_seed)])
            mean = delta.mean((0, 1)) * 100
            wins = sum(metrics(a)['acc05'] > metrics(b)['acc05'] for a, b in zip(rows_by_seed, baseline_by_seed))
            stats = {'mean_delta_percentage_points': dict(zip(('acc05', 'acc07', 'miou'), mean.tolist())),
                     'acc05_improved_seeds': wins,
                     'consistent_improvement': bool(mean[0] > 0 and mean[1] >= 0 and mean[2] >= 0 and wins >= 2),
                     'image_bootstrap': bootstrap_image_deltas([r['image'] for r in rows_by_seed[0]], delta.mean(0)),
                     'uncertainty': 'paired images conditional on these three trained seeds'}
            if variant == 'text_local':
                local = [predictions['local', seed, split] for seed in (13, 14, 15)]
                mechanism = np.asarray([difference(a, b) for a, b in zip(rows_by_seed, local)])
                stats['text_conditioning_ablation'] = bootstrap_image_deltas(
                    [r['image'] for r in rows_by_seed[0]], mechanism.mean(0))
            result['three_seed'][f'{variant}/{split}'] = stats
    return result
