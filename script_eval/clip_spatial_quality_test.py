"""Report held-out results for checkpoints fixed by validation, without training.

Run this file directly: imports are taken from the training source snapshot.
The prepare step commits every checkpoint and split digest before test inference.
"""
import argparse
import gc
import json
from pathlib import Path
import sys


def dependencies(root):
    sys.path.insert(0, str(root / 'source'))
    from script_train.clip_spatial_quality_control import runtime
    return runtime


def prepare(root, output):
    r = dependencies(root)
    from script_train.clip_spatial_quality_control.control import verify_snapshot
    if output.exists():
        raise FileExistsError(output)
    registration = verify_snapshot(root)
    completed = json.loads((root / 'report.json').read_text())
    a = r.recipe(output, 'c1_continue')
    cases = {'original_c1': {'variant': 'c1_continue',
             'checkpoint': a.exchange_base_checkpoint, 'epoch': 100,
             'validation': r.reference(13)['validation'], 'full_base_checkpoint': True}}
    for variant in r.VARIANTS:
        directory = root / 'explore' / (variant + '_seed13')
        selected = json.loads((directory / 'best.json').read_text())
        if json.dumps(selected, sort_keys=True) != json.dumps(completed['runs'][variant], sort_keys=True):
            raise ValueError('Completed report and selected checkpoint metadata differ')
        cases[variant] = {'variant': variant, 'checkpoint': str(directory / 'best_checkpoint.pth'),
                          'epoch': selected['epoch'], 'validation': selected,
                          'full_base_checkpoint': False}
    for case in cases.values():
        case['sha256'] = r.sha256(case['checkpoint'])
    annotations = {split: Path(a.split_root) / a.dataset / f'{a.dataset}_{split}.pth'
                   for split in ('val', 'test')}
    r.write_json(output / 'evaluation_plan.json', {
        'training_root': str(root), 'source_manifest_sha256': registration['provenance']['source_sha256'],
        'evaluation_script_sha256': r.sha256(Path(__file__)),
        'selection': 'First strict maximum validation Acc@0.5; no test-dependent selection',
        'checkpoints': cases, 'test_previously_observed_for_original_c1': True,
        'annotations': {k: {'path': str(p), 'sha256': r.sha256(p)} for k, p in annotations.items()},
        'protocol': {'world_size': 2, 'batch_size_per_gpu': 12, 'grad_context': 'torch.no_grad',
                     'optimizer_created': False, 'ground_truth_in_forward': False,
                     'splits': ['val', 'test'], 'bootstrap_unit': 'image', 'bootstrap_repeats': 5000}})
    r.write_json(output / 'status.json', {'phase': 'prepared', 'completed': [], 'error': None})
    print(json.dumps({'prepared': str(output), 'cases': list(cases)}), flush=True)


def run(root, output):
    r = dependencies(root)
    import torch
    import torch.distributed as dist
    from torch.utils.data import DataLoader, DistributedSampler
    from script_train.clip_reobserve_control.data import NativeDataset
    from script_train.clip_soft_control.evaluate import IndexedDataset, summarize
    from utils import misc
    from utils.box_utils import bbox_iou, xywh2xyxy

    plan = json.loads((output / 'evaluation_plan.json').read_text())
    if (r.sha256(root / 'source_manifest.json') != plan['source_manifest_sha256']
            or r.sha256(Path(__file__)) != plan['evaluation_script_sha256']):
        raise ValueError('Evaluation source changed after checkpoint registration')
    for source in plan['annotations'].values():
        if r.sha256(source['path']) != source['sha256']:
            raise ValueError('Evaluation annotations changed')
    args = r.recipe(output, 'c1_continue')
    misc.init_distributed_mode(args)
    rank, world = misc.get_rank(), misc.get_world_size()
    if world != 2:
        raise ValueError('Evaluation requires the registered two-GPU protocol')
    torch.set_num_threads(4)
    device = torch.device('cuda', args.gpu)
    done = []

    for name, case in plan['checkpoints'].items():
        if r.sha256(case['checkpoint']) != case['sha256']:
            raise ValueError('A selected checkpoint changed')
        state = torch.load(case['checkpoint'], map_location='cpu', weights_only=False)
        if state['epoch'] != case['epoch']:
            raise ValueError('Selected checkpoint epoch differs from its validation metadata')
        a = r.recipe(output, 'c1_continue') if case['full_base_checkpoint'] else state['args']
        a.gpu, a.rank, a.world_size, a.distributed = args.gpu, rank, world, True
        r.seed_all(a.seed + rank)
        model = r.build(a)
        r.initialize(model, a)
        if not case['full_base_checkpoint']:
            r.restore_delta(model, a, state)
        del state
        model.to(device).eval()
        wrapped = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],
                                                           find_unused_parameters=True)
        captured = {}
        if model.refiner is not None:
            original_decode = model.refiner.decode
            def decode(*values, **kwargs):
                prediction, auxiliary = original_decode(*values, **kwargs)
                captured.clear()
                captured.update(auxiliary)
                return prediction, auxiliary
            model.refiner.decode = decode

        for split in ('val', 'test'):
            path = output / name / (split + '.json')
            if path.exists():
                raise FileExistsError(path)
            if rank == 0:
                r.write_json(output / 'status.json', {'phase': 'evaluating', 'case': name,
                    'split': split, 'completed': done, 'error': None})
            data = NativeDataset(split, a)
            indexed = IndexedDataset(data)
            sampler = DistributedSampler(indexed, shuffle=False)
            loader = DataLoader(indexed, batch_size=12, sampler=sampler, num_workers=2,
                                collate_fn=collate)
            local = []
            with torch.no_grad():
                for batch_number, (indices, (images, phrases, targets, _)) in enumerate(loader):
                    targets = targets.to(device)
                    prediction = wrapped(images.to(device), phrases)[0]
                    values = bbox_iou(xywh2xyxy(prediction).clamp(0, 1), xywh2xyxy(targets))
                    coarse = captured.get('reobserve_coarse', prediction)
                    coarse_iou = bbox_iou(xywh2xyxy(coarse).clamp(0, 1), xywh2xyxy(targets))
                    proposals = captured.get('reobserve_proposals')
                    proposal_iou = ([bbox_iou(xywh2xyxy(proposals[:, i]).clamp(0, 1),
                                             xywh2xyxy(targets)) for i in range(2)]
                                    if proposals is not None else None)
                    for j, index in enumerate(indices):
                        ann = data.images[index]
                        row = {'index': index, 'image': str(ann[0]), 'phrase': phrases[j],
                            'target_xywh': targets[j].tolist(), 'pred_xywh': prediction[j].tolist(),
                            'iou': float(values[j]), 'coarse_iou': float(coarse_iou[j]),
                            'size_label': str(ann[5]).lower(), 'weak': str(ann[4]) in ('1', '2')}
                        if proposal_iou is not None:
                            row.update(proposal_ious=[float(x[j]) for x in proposal_iou],
                                predicted_gain=captured['spatial_predicted_gain'][j].tolist(),
                                local_weights=captured['spatial_weights'][j].tolist(),
                                entropy=captured['spatial_entropy'][j].tolist())
                        local.append(row)
                    if rank == 0 and (batch_number + 1) % 30 == 0:
                        print(f'{name} {split}: rank0 {len(local)}/{len(sampler)}', flush=True)
            gathered = [None] * world
            dist.all_gather_object(gathered, local)
            if rank == 0:
                all_rows = [row for part in gathered for row in part]
                rows = deduplicate(all_rows, len(data))
                summary = summarize(rows)
                if split == 'val':
                    expected = case['validation']
                    for current, key in [('acc_05', 'accu'), ('acc_07', 'accu_07'), ('miou', 'miou')]:
                        if abs(summary[current] - expected[key]) > (1e-5 if key == 'miou' else 1e-12):
                            raise AssertionError(f'Validation did not reproduce: {name} {key}')
                r.write_json(path, {'source': case, 'split': split, 'summary': summary,
                    'protocol': {**plan['protocol'], 'duplicate_padding_removed': len(all_rows) - len(rows),
                                 'validation_reproduced_before_test': True}, 'samples': rows})
                done.append(name + '/' + split)
                print(json.dumps({'case': name, 'split': split, **summary}), flush=True)
            dist.barrier()
            del loader, data, indexed
        if model.refiner is not None:
            model.refiner.decode = original_decode
        del wrapped, model
        captured.clear()
        gc.collect()
        torch.cuda.empty_cache()
    if rank == 0:
        for case in plan['checkpoints'].values():
            if r.sha256(case['checkpoint']) != case['sha256']:
                raise ValueError('Evaluation changed a checkpoint')
        r.write_json(output / 'status.json', {'phase': 'complete', 'completed': done,
                     'checkpoints_unchanged': True, 'error': None})
    dist.barrier()
    dist.destroy_process_group()


def collate(rows):
    from script_train.clip_reobserve_control.data import collate_native
    indices, values = zip(*rows)
    return indices, collate_native(values)


def deduplicate(rows, count):
    by_index = {}
    for row in rows:
        index = row['index']
        if index in by_index:
            prior = by_index[index]
            if (prior['image'] != row['image'] or prior['phrase'] != row['phrase']
                    or abs(prior['iou'] - row['iou']) > 1e-5):
                raise ValueError('Distributed sampler duplicate predictions disagree')
        by_index[index] = row
    if set(by_index) != set(range(count)):
        raise ValueError('Evaluation has missing or unexpected sample indices')
    return [by_index[index] for index in range(count)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run'))
    parser.add_argument('--root', type=Path, default=Path('output_clip_lavs_control/2026-10-27_spatial_quality'))
    parser.add_argument('--output', type=Path, default=Path('output_clip_analysis/2026-10-28_spatial_quality_test'))
    cli = parser.parse_args()
    getattr(sys.modules[__name__], cli.action)(cli.root.resolve(), cli.output.resolve())
