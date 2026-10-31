"""Real two-rank checks for inherited initialization and trainable routing."""
import gc
import hashlib
import math
import time

from .runtime import (VARIANTS, recipe, initialize, seed_all, optimizer_for,
                      rng_state, restore_rng, atomic_checkpoint, write_json)


def inherited_digest(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        if not name.startswith('local_modal_exchange.'):
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def run(root):
    import torch
    import torch.distributed as dist
    from torch.utils.data import DataLoader, Subset, DistributedSampler, BatchSampler
    from datasets import build_dataset
    from models import build_model
    from utils import misc
    from engine import train_one_epoch
    args = recipe(root/'preflight')
    misc.init_distributed_mode(args)
    torch.set_num_threads(4)
    rank = misc.get_rank()
    if misc.get_world_size() != 2:
        raise AssertionError('Two ranks required')
    device = torch.device('cuda', args.gpu)
    seed_all(13+rank)
    baseline = build_model(recipe(root/'preflight', backend='legacy'))
    inherited = initialize(baseline, args.retrain)
    baseline_count = sum(p.numel() for p in baseline.parameters())
    expected_digest = inherited_digest(baseline)
    del baseline
    gc.collect()
    data = build_dataset('val', args)
    reference = misc.collate_fn([data[i] for i in range(rank*12, (rank+1)*12)])
    def fresh_batch():
        images, phrases, _, _ = reference
        return misc.NestedTensor(images.tensors.clone().to(device), images.mask.clone().to(device)), phrases
    expected = None
    timings = {}
    for variant in VARIANTS:
        seed_all(13+rank)
        model_args = recipe(root/'preflight', variant)
        model = build_model(model_args)
        loading = initialize(model, args.retrain, inherited['missing_keys'], inherited['removed_or_mapped_keys'])
        if inherited_digest(model) != expected_digest:
            raise AssertionError('Inherited weights differ from matched legacy initialization')
        model.to(device).eval()
        with torch.no_grad():
            result = model(*fresh_batch())
            if len(result) != 5 or not torch.isfinite(result[0]).all():
                raise AssertionError('Evaluation API or finite predictions changed')
            if expected is None:
                expected = result[0].detach().clone()
            delta = (expected-result[0]).abs().max().item()
            if delta > 1e-6:
                raise AssertionError('Text-on/off initial predictions differ')
            torch.cuda.synchronize()
            tick = time.monotonic()
            for _ in range(3):
                model(*fresh_batch())
            torch.cuda.synchronize()
        added = sum(p.numel() for p in model.parameters()) - baseline_count
        if added > 2_000_000:
            raise AssertionError('Local gate exceeds two million additional parameters')
        timings[variant] = {'max_initial_error': delta, 'batch12_seconds': (time.monotonic()-tick)/3,
                           'net_added_parameters': added, 'initialization': loading}
        del result
        if variant != VARIANTS[-1]:
            del model
            gc.collect()
            torch.cuda.empty_cache()
    # Two optimizer updates reveal gradients behind the initially zero gate head.
    training = Subset(build_dataset('train', model_args), list(range(192)))
    sampler = DistributedSampler(training, shuffle=True)
    loader = DataLoader(training, batch_sampler=BatchSampler(sampler, 12, drop_last=True),
                        num_workers=2, collate_fn=misc.collate_fn)
    wrapped = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
    optimizer, scheduler = optimizer_for(model, model_args)
    watch = {name: p.detach().clone() for name, p in model.named_parameters()
             if ('.lora_B.' in name and ('layers.0.' in name or 'layers.11.' in name))
             or (name.startswith('local_modal_exchange.') and name.endswith('text_proj.weight'))}
    gradient_l1, hooks = {}, []
    for name, parameter in model.named_parameters():
        if name in watch:
            def record(gradient, name=name):
                gradient_l1[name] = gradient_l1.get(name, 0.) + gradient.detach().abs().sum().item()
            hooks.append(parameter.register_hook(record))
    torch.cuda.reset_peak_memory_stats()
    stats = train_one_epoch(model_args, wrapped, loader, optimizer, device, 0, 1.)
    scheduler.step()
    for hook in hooks:
        hook.remove()
    sums = {branch: sum(value for name, value in gradient_l1.items() if branch in name)
            for branch in ('lora_rgb', 'lora_ir', 'text_proj')}
    if not all(value > 0 and math.isfinite(value) for value in sums.values()):
        raise AssertionError(f'Both LoRA adapters and text gates require gradients: {sums}')
    updated = {branch: [name for name, before in watch.items()
                       if branch in name and not torch.equal(before, dict(model.named_parameters())[name])]
               for branch in sums}
    if not all(updated.values()) or not all(torch.isfinite(p).all() for p in model.parameters()):
        raise AssertionError('Watched parameters did not update or weights are nonfinite')
    steps = {int(s['step'].item()) for s in optimizer.state.values() if 'step' in s}
    if steps != {2}:
        raise AssertionError(f'Accumulation should perform two updates: {steps}')
    peak = torch.cuda.max_memory_allocated()
    per_rank = [None]*2
    state = rng_state()
    dist.all_gather_object(per_rank, state)
    probe = root/'preflight/recovery_probe.pth'
    if rank == 0:
        atomic_checkpoint(probe, {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                                 'lr_scheduler': scheduler.state_dict(), 'rng_by_rank': per_rank})
    dist.barrier()
    checkpoint = torch.load(probe, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'], strict=True)
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['lr_scheduler'])
    restore_rng(checkpoint['rng_by_rank'][rank])
    a = torch.rand(8, device=device)
    restore_rng(state)
    if not torch.equal(a, torch.rand(8, device=device)):
        raise AssertionError('Per-rank RNG restoration differs')
    if any(not torch.equal(before, model.state_dict()[name].cpu()) for name, before in checkpoint['model'].items()):
        raise AssertionError('Strict model checkpoint restoration differs')
    del checkpoint
    ranks = [None]*2
    dist.all_gather_object(ranks, {'rank': rank, 'updated_parameters': updated,
                                 'train_peak_memory_bytes': peak, 'gradient_l1': sums,
                                 'optimizer_updates': 2, 'checkpoint_rng_restored': True})
    if rank == 0:
        write_json(root/'preflight.json', {'passed': True, 'baseline_parameters': baseline_count,
                   'inherited_missing_keys': inherited['missing_keys'],
                   'inherited_extra_keys': inherited['removed_or_mapped_keys'], 'variants': timings,
                   'two_gpu_update': ranks, 'training_metrics': stats})
        probe.unlink()
    dist.barrier()
    dist.destroy_process_group()
