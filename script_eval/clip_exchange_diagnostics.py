"""Validation-only sensitivity checks on frozen external exchange checkpoints.

These interventions do not train or select a new model. Removing an attention
constraint at inference introduces distribution shift; results are diagnostic,
not an estimate of the performance of a retrained replacement architecture.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys
import types


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


@contextmanager
def intervention(exchange, mode):
    import torch
    if mode not in ('baseline', 'unit_gate', 'global_graph', 'zero_text'):
        raise ValueError('Unknown diagnostic intervention')
    forwards = [gate.forward for gate in exchange.gates]
    graphs = [getattr(exchange, f'graph_{i}').clone() for i in range(len(exchange.gates))]
    try:
        if mode == 'global_graph':
            for i in range(len(exchange.gates)):
                getattr(exchange, f'graph_{i}').fill_(True)
        elif mode == 'unit_gate':
            def unit(self, recipient, message, text, use_text=True):
                return torch.ones_like(recipient[..., :1])
            for gate in exchange.gates:
                gate.forward = types.MethodType(unit, gate)
        elif mode == 'zero_text':
            for gate, original in zip(exchange.gates, forwards):
                def zero(self, recipient, message, text, use_text=True, original=original):
                    return original(recipient, message, text, use_text=False)
                gate.forward = types.MethodType(zero, gate)
        yield
    finally:
        for i, (gate, original, graph) in enumerate(zip(exchange.gates, forwards, graphs)):
            gate.forward = original
            getattr(exchange, f'graph_{i}').copy_(graph)


def compare_identity(rows, reference):
    if len(rows) != len(reference):
        raise AssertionError('Incomplete prediction set')
    for a, b in zip(rows, reference):
        if any(a[k] != b[k] for k in ('index', 'image', 'phrase', 'target_xywh')):
            raise AssertionError('Annotation identities differ')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    cli = parser.parse_args()
    experiment, output = cli.experiment.resolve(), cli.output_root.resolve()
    if (output/'report.json').exists():
        raise FileExistsError('Completed diagnostics exist; use a new output directory')
    preregistration = json.loads((experiment/'preregistration.json').read_text())
    source = experiment/'source'
    source_manifest = json.loads((experiment/'source_manifest.json').read_text())
    if sha256(experiment/'source_manifest.json') != preregistration['source_manifest_sha256']:
        raise ValueError('Frozen source manifest changed')
    for name, expected in source_manifest.items():
        if sha256(source/name) != expected:
            raise ValueError(f'Frozen source changed: {name}')
    sys.path.insert(0, str(source))
    import numpy as np
    import torch
    import torch.distributed as dist
    from torch.utils.data import DataLoader, DistributedSampler
    from datasets import build_dataset
    from models import build_model
    from utils import misc
    from utils.box_utils import bbox_iou, xywh2xyxy
    from script_train.clip_lavs_control.evaluate import IndexedDataset, collate_indexed, summarize
    from script_train.clip_structure.report import image_paired_bootstrap
    selected = json.loads((experiment/'selected_models.json').read_text())
    c1_item = next(x for x in preregistration['references'] if x['seed'] == 13)
    c1_path = Path(c1_item['checkpoint']).parents[1]/'evaluation/c1_seed13/val.json'
    c1_rows = json.loads(c1_path.read_text())['samples']
    result = {'purpose': 'validation-only sensitivity, not retrained model selection',
              'test_used': False, 'source_manifest_sha256': preregistration['source_manifest_sha256'],
              'diagnostic_script_sha256': sha256(__file__), 'models': {}}
    initialized = False
    for item in selected:
        if sha256(item['checkpoint']) != item['sha256']:
            raise ValueError('Frozen checkpoint changed')
        checkpoint = torch.load(item['checkpoint'], map_location='cpu', weights_only=False)
        args = checkpoint['args']
        if not initialized:
            misc.init_distributed_mode(args)
            rank = misc.get_rank()
            if misc.get_world_size() != 2:
                raise ValueError('Match the original two-rank evaluation protocol')
            initialized = True
        else:
            args.gpu = int(__import__('os').environ['LOCAL_RANK'])
        torch.set_num_threads(4)
        torch.manual_seed(args.seed + rank)
        np.random.seed(args.seed + rank)
        device = torch.device('cuda', args.gpu)
        model = build_model(args)
        model.load_state_dict(checkpoint['model'], strict=True)
        del checkpoint
        model.to(device).eval()
        wrapped = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        dataset = build_dataset('val', args)
        annotations = torch.load(Path(args.split_root)/args.dataset/f'{args.dataset}_val.pth', map_location='cpu', weights_only=False)
        sampler = DistributedSampler(IndexedDataset(dataset), shuffle=False)
        loader = DataLoader(IndexedDataset(dataset), batch_size=args.batch_size, sampler=sampler,
                            num_workers=args.num_workers, collate_fn=collate_indexed)
        reference = json.loads((experiment/'evaluation'/item['name']/'val.json').read_text())['samples']
        modes = ('baseline', 'unit_gate', 'global_graph', 'zero_text') if item['variant'] == 'text_local' else ('baseline', 'unit_gate', 'global_graph')
        if rank == 0:
            write_json(output/'registration.json', {'source_manifest_sha256': preregistration['source_manifest_sha256'],
                       'checkpoints': selected, 'test_used': False, 'modes': {'local': modes[:3], 'text_local': ['baseline', 'unit_gate', 'global_graph', 'zero_text']}})
        for mode in modes:
            output_path = output/item['name']/mode/'val.json'
            if output_path.exists():
                raise FileExistsError(f'Refusing to overwrite {output_path}')
            local_rows = []
            with intervention(model.local_modal_exchange, mode), torch.no_grad():
                for indices, (images, phrases, targets, _) in loader:
                    images, targets = images.to(device), targets.to(device)
                    boxes = wrapped(images, phrases)[0]
                    values = bbox_iou(xywh2xyxy(boxes).clamp(0, 1), xywh2xyxy(targets))
                    for index, phrase, box, target, value in zip(indices, phrases, boxes.cpu(), targets.cpu(), values.cpu()):
                        local_rows.append({'index': index, 'image': str(dataset.images[index][0]), 'phrase': phrase,
                                           'pred_xywh': box.tolist(), 'target_xywh': target.tolist(), 'iou': value.item()})
            gathered = [None]*2
            dist.all_gather_object(gathered, local_rows)
            if rank == 0:
                by_index = {row['index']: row for rows in gathered for row in rows}
                rows = [by_index[index] for index in range(len(dataset))]
                compare_identity(rows, reference)
                if mode == 'baseline':
                    error = np.max(np.abs(np.array([x['pred_xywh'] for x in rows]) - np.array([x['pred_xywh'] for x in reference])))
                    if error > 1e-6:
                        raise AssertionError(f'Frozen baseline reproduction differs by {error}')
                groups = {label: summarize([row for row, ann in zip(rows, annotations) if predicate(ann)])
                          for label, predicate in {'weak': lambda x: str(x[4]) in ('1', '2'),
                                                   'small': lambda x: str(x[5]).lower() == 'small'}.items()}
                summary = {'metrics': summarize(rows), 'groups': groups,
                           'paired_frozen': image_paired_bootstrap(rows, reference),
                           'paired_c1': image_paired_bootstrap(rows, c1_rows)}
                write_json(output_path, {'checkpoint_sha256': item['sha256'], 'epoch': item['epoch'],
                           'mode': mode, 'split': 'val', 'summary': summary, 'samples': rows})
                result['models'].setdefault(item['name'], {})[mode] = summary
                print(json.dumps({'model': item['name'], 'mode': mode, **summary['metrics']}, ensure_ascii=False), flush=True)
            dist.barrier()
        del wrapped, model, loader, dataset
        torch.cuda.empty_cache()
    if rank == 0:
        write_json(output/'report.json', result)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
