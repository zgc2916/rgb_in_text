"""Validation-only intervention audit of completed native re-observation runs."""
import argparse
import gc
import json
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.utils.data import DataLoader, DistributedSampler

from script_train.clip_reobserve_control import runtime as r
from script_train.clip_reobserve_control.data import NativeDataset, NativeBatch, collate_native
from script_train.clip_soft_control.evaluate import IndexedDataset, summarize
from utils import misc
from utils.box_utils import bbox_iou, xywh2xyxy


def collate(rows):
    indices, values = zip(*rows)
    return indices, collate_native(values)


def blur_native(images):
    native = images.native.clone()
    for i, size in enumerate(images.sizes):
        h, w = (int(x) for x in size)
        pixels = native[i:i + 1, :, :h, :w].float()
        pixels = F.interpolate(pixels, (max(1, h // 8), max(1, w // 8)), mode='area')
        native[i:i + 1, :, :h, :w] = F.interpolate(pixels, (h, w), mode='bilinear', align_corners=False).round().clamp(0, 255).byte()
    return NativeBatch(images.tensors, images.mask, native, images.sizes, images.rectangles)


def run(root, output):
    if output.exists():
        raise FileExistsError(output)
    args = r.recipe(output)
    misc.init_distributed_mode(args)
    torch.set_num_threads(4)
    rank, world = misc.get_rank(), misc.get_world_size()
    if world != 2:
        raise ValueError('Match the registered two-GPU validation protocol')
    device = torch.device('cuda', args.gpu)
    cases = [('fixed_best', 'fixed_local', 'trained_best_checkpoint.pth'),
             ('consistent_best', 'text_consistent', 'trained_best_checkpoint.pth'),
             ('consistent_last', 'text_consistent', 'checkpoint.pth')]
    report = {'test_used': False, 'purpose': 'intervention audit, not new checkpoint selection', 'cases': {}}
    for name, variant, filename in cases:
        path = root / 'explore' / (variant + '_seed13') / filename
        state = torch.load(path, map_location='cpu', weights_only=False)
        a = r.recipe(output, variant)
        a.gpu, a.rank, a.distributed = args.gpu, rank, True
        r.seed_all(13 + rank)
        model = r.build(a)
        r.initialize(model, a)
        r.restore_delta(model, a, state)
        model.to(device).eval()
        data = NativeDataset('val', a)
        indexed = IndexedDataset(data)
        loader = DataLoader(indexed, batch_size=12, sampler=DistributedSampler(indexed, shuffle=False),
                            num_workers=2, collate_fn=collate)
        original_decode = model.refiner.decode
        captured = {}
        def decode(*values, **kwargs):
            final, auxiliary = original_decode(*values, **kwargs)
            captured.update(auxiliary)
            return final, auxiliary
        model.refiner.decode = decode
        case = {'checkpoint': str(path), 'sha256': r.sha256(path), 'epoch': state['epoch'], 'modes': {}}
        for mode in ('full', 'coarse_only', 'blur_local'):
            local = []
            with torch.no_grad():
                for indices, (images, phrases, target, _) in loader:
                    images, target = images.to(device), target.to(device)
                    if mode == 'blur_local':
                        images = blur_native(images)
                    prediction = model(images, phrases)[0]
                    coarse = captured['reobserve_coarse']
                    if mode == 'coarse_only':
                        prediction = coarse
                    values = bbox_iou(xywh2xyxy(prediction).clamp(0, 1), xywh2xyxy(target))
                    coarse_values = bbox_iou(xywh2xyxy(coarse).clamp(0, 1), xywh2xyxy(target))
                    proposal = captured['reobserve_proposals']
                    proposal_iou = [bbox_iou(xywh2xyxy(proposal[:, i]).clamp(0, 1), xywh2xyxy(target)) for i in range(2)]
                    for j, index in enumerate(indices):
                        local.append({'index': index, 'image': data.images[index][0], 'iou': float(values[j]),
                            'coarse_iou': float(coarse_values[j]), 'pred_xywh': prediction[j].tolist(),
                            'coarse_xywh': coarse[j].tolist(), 'accept': float(captured['reobserve_accept_logits'][j].sigmoid()),
                            'proposal_ious': [float(v[j]) for v in proposal_iou]})
            gathered = [None] * world
            dist.all_gather_object(gathered, local)
            if rank == 0:
                by_index = {row['index']: row for part in gathered for row in part}
                rows = [by_index[i] for i in range(len(data))]
                quality = torch.tensor([x['iou'] for x in rows])
                coarse_quality = torch.tensor([x['coarse_iou'] for x in rows])
                best_local = torch.tensor([max(x['proposal_ious']) for x in rows])
                summary = {**summarize(rows), 'mean_accept': sum(x['accept'] for x in rows) / len(rows),
                    'gained_acc05_vs_same_model_coarse': int(((quality >= .5) & (coarse_quality < .5)).sum()),
                    'lost_acc05_vs_same_model_coarse': int(((quality < .5) & (coarse_quality >= .5)).sum()),
                    'local_best_improves_iou_gt001': int((best_local > coarse_quality + .01).sum()),
                    'both_local_worse_iou_gt001': int((best_local < coarse_quality - .01).sum()),
                    'mean_iou_delta_vs_same_model_coarse': float((quality - coarse_quality).mean())}
                case['modes'][mode] = summary
                r.write_json(output / name / (mode + '.json'), {'summary': summary, 'samples': rows})
                print(name, mode, json.dumps(summary), flush=True)
                if mode == 'full' and (abs(summary['acc_05'] - state['val_accu']) > 1e-12
                                        or abs(summary['miou'] - state['val_miou']) > 1e-5):
                    raise AssertionError('Frozen validation result did not reproduce')
            dist.barrier()
        if rank == 0:
            report['cases'][name] = case
            r.write_json(output / 'report.json', report)
        model.refiner.decode = original_decode
        del model, state, loader, data, indexed
        gc.collect()
        torch.cuda.empty_cache()
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('output_clip_lavs_control/2026-10-27_native_reobserve'))
    parser.add_argument('--output', type=Path, default=Path('output_clip_analysis/2026-10-27_reobserve_quality'))
    cli = parser.parse_args()
    run(cli.root.resolve(), cli.output.resolve())
