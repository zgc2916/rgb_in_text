"""Paired image-cluster statistics for preregistered structure checkpoints."""
import json
from pathlib import Path

import numpy as np


def metrics(rows):
    values = np.asarray([row['iou'] for row in rows])
    return {'count': len(rows), 'acc05': float((values >= .5).mean()),
            'acc07': float((values >= .7).mean()), 'miou': float(values.mean())}


def image_paired_bootstrap(candidate, reference, repeats=5000, seed=13):
    if len(candidate) != len(reference) or not candidate:
        raise ValueError('Need two complete paired prediction sets')
    deltas = []
    for a, b in zip(candidate, reference):
        if any(a[k] != b[k] for k in ('index', 'image', 'phrase', 'target_xywh')):
            raise ValueError('Prediction pairs differ in annotation identity')
        delta = np.array([float(a['iou'] >= .5) - float(b['iou'] >= .5),
                          float(a['iou'] >= .7) - float(b['iou'] >= .7), a['iou'] - b['iou']])
        deltas.append(delta)
    return bootstrap_image_deltas([row['image'] for row in candidate], np.asarray(deltas), repeats, seed)


def bootstrap_image_deltas(images, deltas, repeats=5000, seed=13):
    groups = {}
    for image, delta in zip(images, deltas):
        total, count = groups.setdefault(image, (np.zeros(3), 0))
        groups[image] = (total + delta, count + 1)
    # Draw images; retain all expressions within each sampled image. Weight
    # aggregate accuracy by expression count to match the reported estimand.
    sums = np.stack([x[0] for x in groups.values()])
    counts = np.asarray([x[1] for x in groups.values()])
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(repeats):
        indices = rng.integers(len(groups), size=len(groups))
        boot.append(sums[indices].sum(0) / counts[indices].sum())
    lower, upper = np.quantile(np.asarray(boot), [.025, .975], axis=0)
    estimate = sums.sum(0) / counts.sum()
    return {'unit': 'image', 'images': len(groups), 'expressions': len(images), 'repeats': repeats,
            'delta_percentage_points': dict(zip(('acc05', 'acc07', 'miou'), (estimate * 100).tolist())),
            'ci95_percentage_points': {k: [float(lo * 100), float(hi * 100)]
                for k, lo, hi in zip(('acc05', 'acc07', 'miou'), lower, upper)}}


def build_report(root):
    root = Path(root)
    selected = json.loads((root / 'selected_models.json').read_text())
    from .run_control import sha256
    evaluations, result = {}, {'selection': 'first strict maximum validation Acc05', 'runs': {}}
    result['implementation_audit'] = json.loads((root / 'preflight.json').read_text())
    result['nearest_work'] = {
        'ViT-Adapter': 'https://arxiv.org/abs/2205.08534',
        'BAT': 'https://ojs.aaai.org/index.php/AAAI/article/view/27852',
        'TFDet': 'https://arxiv.org/abs/2305.16580',
        'Dynamic MDETR': 'https://arxiv.org/abs/2209.13959',
        'SABL': 'https://arxiv.org/abs/1912.04260',
        'DUALVISION': 'https://arxiv.org/abs/2604.18829',
        'Thermo-VL': 'https://arxiv.org/abs/2605.21882',
        'Bridging the RGB-IR Gap': 'https://arxiv.org/abs/2604.11234'}
    result['novelty_status'] = ('Hypothesis: detail supports both pre-LAVS TIR correction and independent-memory boundary reading. '
                               'CNN injection, gating and side queries are prior techniques. Budget-matched mechanism '
                               'retraining is required before claiming novelty and is outside this first eight-run suite.')
    for item in selected:
        name = f"{item['variant']}_seed{item['seed']}"
        if sha256(item['checkpoint']) != item['sha256']:
            raise AssertionError('Preregistered checkpoint changed')
        evaluations[name] = {split: json.loads((root / 'evaluation' / name / f'{split}.json').read_text())
                             for split in ('val', 'test')}
        if any(v['source']['epoch'] != item['epoch'] for v in evaluations[name].values()):
            raise AssertionError('Evaluation used a different epoch')
        result['runs'][name] = {'epoch': item['epoch'], 'sha256': item['sha256'],
            'validation_same_checkpoint_groups': item['validation'],
            **{s: metrics(v['samples']) for s, v in evaluations[name].items()}}
    result['paired'] = {}
    for name, splits in evaluations.items():
        if name.startswith('c1_'):
            continue
        reference = evaluations['c1_' + name.split('_')[-1]]
        result['paired'][name] = {s: image_paired_bootstrap(v['samples'], reference[s]['samples'])
                                  for s, v in splits.items()}
        # Persist every improvement/regression for review, not just aggregate peaks.
        from .run_control import write_json
        for s, v in splits.items():
            changes = [{'index': a['index'], 'image': a['image'], 'phrase': a['phrase'],
                        'reference_iou': b['iou'], 'candidate_iou': a['iou'], 'delta_iou': a['iou'] - b['iou']}
                       for a, b in zip(v['samples'], reference[s]['samples'])]
            write_json(root / 'changes' / name / f'{s}.json', changes)
    result['three_seed'] = {}
    for split in ('val', 'test'):
        c1 = [result['runs'][f'c1_seed{s}'][split] for s in (13, 14, 15)]
        joint = [result['runs'][f'joint_seed{s}'][split] for s in (13, 14, 15)]
        delta = {k: float(np.mean([b[k] - a[k] for a, b in zip(c1, joint)]) * 100)
                 for k in ('acc05', 'acc07', 'miou')}
        wins = sum(b['acc05'] > a['acc05'] for a, b in zip(c1, joint))
        result['three_seed'][split] = {'mean_delta_percentage_points': delta, 'acc05_improved_seeds': wins,
            'first_round_success': delta['acc05'] > 0 and delta['acc07'] >= 0 and delta['miou'] >= 0 and wins >= 2,
            'continued_investment_target_reached': delta['acc05'] >= 1}
        per_seed = []
        for seed in (13, 14, 15):
            a = evaluations[f'c1_seed{seed}'][split]['samples']
            b = evaluations[f'joint_seed{seed}'][split]['samples']
            # Validate annotation order before averaging across fixed seeds.
            image_paired_bootstrap(b, a, repeats=1)
            per_seed.append([[float(y['iou'] >= .5) - float(x['iou'] >= .5),
                              float(y['iou'] >= .7) - float(x['iou'] >= .7), y['iou'] - x['iou']]
                             for x, y in zip(a, b)])
        averaged = np.asarray(per_seed).mean(0)
        result['three_seed'][split]['paired_image_bootstrap'] = bootstrap_image_deltas(
            [r['image'] for r in evaluations['c1_seed13'][split]['samples']], averaged)
        result['three_seed'][split]['bootstrap_conditioning'] = 'fixed three trained seeds; sampling uncertainty over images'
    # Test subgroup summaries at the same frozen checkpoint, using released labels.
    import torch
    recipes = json.loads((root / 'preregistration.json').read_text())['recipes']
    annotation_root = Path(recipes['formal/c1_seed13']['split_root']) / 'rgbtvg_flir'
    for split in ('val', 'test'):
        annotations = torch.load(annotation_root / f'rgbtvg_flir_{split}.pth', map_location='cpu', weights_only=False)
        for name, evaluation in evaluations.items():
            rows = evaluation[split]['samples']
            if len(rows) != len(annotations):
                raise AssertionError('Evaluation count does not match annotations')
            groups = {'weak': [r for r, a in zip(rows, annotations) if str(a[4]) in ('1', '2')],
                      'small': [r for r, a in zip(rows, annotations) if str(a[5]).lower() == 'small']}
            result['runs'][name][split]['groups'] = {k: metrics(v) for k, v in groups.items()}
    result['diagnostics'] = {}
    for seed in (13, 14, 15):
        name = f'joint_seed{seed}'
        result['diagnostics'][name] = {}
        for label in ('blur_detail', 'disable_memory'):
            rows = json.loads((root / 'diagnostics' / name / label / 'val.json').read_text())['samples']
            result['diagnostics'][name][label] = image_paired_bootstrap(rows, evaluations[name]['val']['samples'])
    return result


def write_markdown(root, result):
    lines = ['# C1 结构对照首轮报告', '',
        '下表使用首次严格达到验证 Acc@0.5 最大值的同一个 checkpoint；epoch 从 0 计数。探索续训不作为正式性能结论。', '',
        '| 方案 | epoch | Val Acc05 | Val Acc07 | Val mIoU | Test Acc05 | Test Acc07 | Test mIoU |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name, run in result['runs'].items():
        vals = [run[s][m] * 100 for s in ('val', 'test') for m in ('acc05', 'acc07', 'miou')]
        lines.append(f"| {name} | {run['epoch']} | " + ' | '.join(f'{v:.2f}%' for v in vals) + ' |')
    lines += ['', '## 三 seed 与配对统计', '']
    for split, stats in result['three_seed'].items():
        delta = stats['mean_delta_percentage_points']
        lines.append(f"{split}：联合方案平均变化 Acc05 {delta['acc05']:+.3f}、Acc07 {delta['acc07']:+.3f}、mIoU {delta['miou']:+.3f} 个百分点；"
                     f"Acc05 改善 {stats['acc05_improved_seeds']}/3 seed。成功判据：{stats['first_round_success']}。")
        lines.append('95% 区间按图像配对 bootstrap，保留同图全部表达；三 seed 区间以这三次已训练模型为条件，见 report.json。')
        lines.append('')
    lines += ['## 实际参数和预检资源', '',
              '| 配置 | 净增参数 | batch12 推理 ms | 推理峰值 MiB | 更新峰值 MiB |',
              '|---|---:|---:|---:|---:|']
    for row in result['implementation_audit']['ranks'][0]:
        lines.append(f"| {row['variant']} | {row['parameters']['net_added']:,} | {row['inference_ms_batch12']:.2f} | "
                     f"{row['inference_peak_allocated_mib']:.0f} | {row['train_peak_allocated_mib']:.0f} |")
    lines += ['', 'common_c1/common_joint 为公共初始化、全配方可训练预检；detail/boundary/joint 为 C1 冻结探索预检。'
              '耗时是在同一设备每卡 batch12 的 10 次 no_grad 前向均值，显存为 allocated 峰值；各卡值和版本见 preflight.json。', '',
              '同 checkpoint 的弱光/小目标指标、每个 seed 的配对区间、干预诊断见 report.json；逐样本变化保存在 changes/。', '',
              '## 文献近邻与创新边界', '', result['novelty_status'], '']
    lines += [f'- [{title}]({url})' for title, url in result['nearest_work'].items()]
    Path(root, 'report.md').write_text('\n'.join(lines) + '\n')
