"""Small real-data CPU acceptance; not a performance evaluation or GPU test."""
import argparse
import gc
import json
from pathlib import Path
import time

import torch

from . import runtime
from .preflight import frozen_digest


def run(output, examples=8):
    from models import build_model
    from datasets import build_dataset
    from utils import misc
    from utils.loss_utils import trans_vg_loss
    from torch.utils.data import DataLoader, Subset
    torch.set_num_threads(4)
    torch.set_num_interop_threads(2)
    archived = json.loads((runtime.REFERENCE / 'evaluation/c1_seed13/val.json').read_text())
    report = {'device': 'cpu', 'passed': False, 'examples_per_variant': examples,
              'optimizer_updates_per_variant': 2, 'full_validation': False,
              'two_gpu_acceptance_completed': False, 'test_evaluated': False, 'variants': {}}
    paths = ('models/region_anchor_exchange.py', 'models/mmvg_fusion.py', 'utils/loss_utils.py',
             'script_train/clip_anchor_control/runtime.py', 'script_train/clip_anchor_control/cpu_acceptance.py')
    report['source_sha256'] = {name: runtime.sha256(runtime.REPOSITORY / name) for name in paths}
    cpu_baseline = []
    if not 2 <= examples <= len(archived['samples']):
        raise ValueError('Acceptance requires at least two validation examples')
    for variant in runtime.VARIANTS:
        tick = time.monotonic()
        runtime.seed_all(13)
        args = runtime.recipe(output.parent / variant, variant)
        args.device = 'cpu'
        model = build_model(args)
        loading = runtime.initialize(model, args)
        model.eval()
        runtime.assert_scope(model, args)
        dataset = build_dataset('val', args)
        loader = DataLoader(Subset(dataset, range(examples)), batch_size=2,
                            num_workers=0, collate_fn=misc.collate_fn)
        error, same_cpu_error, index = 0., 0., 0
        with torch.no_grad():
            for images, phrases, targets, _ in loader:
                result = model(images, phrases)
                if len(result) != 5:
                    raise AssertionError('Legacy evaluation API changed')
                for phrase, target, box in zip(phrases, targets, result[0]):
                    old = archived['samples'][index]
                    if phrase != old['phrase'] or target.tolist() != old['target_xywh']:
                        raise AssertionError('Validation example identity differs')
                    error = max(error, (box - torch.tensor(old['pred_xywh'])).abs().max().item())
                    if variant == 'c1_continue':
                        cpu_baseline.append(box.clone())
                    else:
                        same_cpu_error = max(same_cpu_error, (box - cpu_baseline[index]).abs().max().item())
                    index += 1
        if same_cpu_error > 1e-6:
            raise AssertionError(f'{variant} initialization differs from the same-device C1: {same_cpu_error}')
        before = frozen_digest(model, args)
        optimizer, scheduler = runtime.optimizer_for(model, args)
        runtime.seed_all(13)
        training = build_dataset('train', args)
        train_loader = DataLoader(Subset(training, range(4)), batch_size=2,
                                  num_workers=0, collate_fn=misc.collate_fn)
        watched = {n: p.detach().clone() for n, p in model.named_parameters()
                   if runtime.selected_name(n, args) and n.endswith('weight')}
        gradients, losses = {}, []
        for images, phrases, targets, mask in train_loader:
            model.train()
            optimizer.zero_grad(set_to_none=True)
            result = model(images, phrases)
            terms = trans_vg_loss(args, result[0], targets, mask, *result[1:], epoch=0)
            if set(terms) != {'loss_bbox', 'loss_giou'}:
                raise AssertionError('Unexpected auxiliary loss in anchor screening')
            loss = sum(terms.values())
            if not torch.isfinite(loss):
                raise AssertionError('Nonfinite grounding loss')
            loss.backward()
            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    if not torch.isfinite(p.grad).all():
                        raise AssertionError(f'Nonfinite gradient: {n}')
                    gradients[n] = gradients.get(n, 0.) + p.grad.abs().sum().item()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.clip_max_norm)
            optimizer.step()
            losses.append(loss.item())
            runtime.assert_scope(model, args)
        scheduler.step()
        if frozen_digest(model, args) != before:
            raise AssertionError('A frozen tensor changed during CPU updates')
        if model.clip.training or model.vl_transformer.training:
            raise AssertionError('Frozen modules must remain in eval mode')
        branches = list(runtime.PROJECTION_PREFIXES)
        if args.exchange_train_new_only:
            branches += [runtime.NEW_PREFIX + 'rgb_output.', runtime.NEW_PREFIX + 'tir_output.']
            if variant != 'anchor_visual':
                branches += [runtime.NEW_PREFIX + 'text_proj.']
        gradient_sums = {prefix: sum(v for n, v in gradients.items() if n.startswith(prefix)) for prefix in branches}
        changed = {prefix: any(not torch.equal(t, dict(model.named_parameters())[n])
                               for n, t in watched.items() if n.startswith(prefix)) for prefix in branches}
        if not all(v > 0 for v in gradient_sums.values()) or not all(changed.values()):
            raise AssertionError(f'Missing required gradient/update: {gradient_sums}, {changed}')
        stats = {'accu': .5, 'accu_07': .3, 'miou': .4}
        state = runtime.delta_record(model, args, 0, stats, .5, {'acceptance': 'cpu_only'})
        state.update(optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict())
        probe = output.parent / f'{variant}_cpu_recovery_probe.pth'
        runtime.atomic_checkpoint(probe, state)
        saved = torch.load(probe, map_location='cpu', weights_only=False)
        with torch.no_grad():
            for n, p in model.named_parameters():
                if runtime.selected_name(n, args):
                    p.add_(1.)
        runtime.restore_delta(model, args, saved)
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        if any(not torch.equal(t, model.state_dict()[n]) for n, t in saved['model_delta'].items()):
            raise AssertionError('Delta serialization/restoration failed')
        if {int(x['step']) for x in optimizer.state.values()} != {2}:
            raise AssertionError('Optimizer recovery lost its update count')
        probe.unlink()
        report['variants'][variant] = {'initialization': loading,
            'maximum_initial_box_error_vs_same_cpu_c1': same_cpu_error,
            'maximum_cpu_vs_archived_gpu_box_difference': error,
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'losses': losses, 'required_gradient_l1': gradient_sums, 'required_weights_updated': changed,
            'frozen_tensors_unchanged': True, 'checkpoint_optimizer_restored': True,
            'seconds': time.monotonic() - tick}
        runtime.write_json(output, report)
        print(json.dumps({'variant': variant, **report['variants'][variant]}), flush=True)
        del state, saved, model, optimizer, scheduler, watched, result, loss, terms, dataset, training
        gc.collect()
    report['passed'] = True
    runtime.write_json(output, report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--examples', type=int, default=8)
    args = parser.parse_args()
    run(args.output.resolve(), args.examples)
