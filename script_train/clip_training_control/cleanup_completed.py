"""Remove verified duplicates and compact completed selective-tuning states."""
import argparse
import gc
import json
from pathlib import Path
import stat
import time

from script_train.clip_lavs_control.runtime import ORIGINAL, sha256, write_json, atomic_checkpoint
from script_train.clip_soft_control.control import living_processes

BASE = ORIGINAL
SUITES = tuple(BASE / 'output_clip_lavs_control' / name for name in (
    '2026-10-18_reliable_exchange_warm', '2026-10-18_clean_joint_control', '2026-10-26_region_anchor'))
STRUCTURE = BASE / 'output_clip_structure_control/2026-10-05_c1_detail_boundary'
TEST_COPY = BASE / 'output_clip_structure_test/2026-10-06_seed14_fixed'
DEFAULT_MANIFEST = BASE / 'output_checkpoint_cleanup/2026-10-27_completed_duplicates/cleanup_manifest.json'
RESUME_FIELDS = {'optimizer', 'lr_scheduler', 'rng_by_rank'}


def info(path):
    path = Path(path)
    if path.is_symlink() or not path.resolve().is_relative_to(BASE.resolve()):
        raise ValueError(f'Cleanup path escapes the workspace: {path}')
    s = path.stat()
    if not stat.S_ISREG(s.st_mode) or s.st_nlink != 1:
        raise ValueError('Only regular files with one link can be removed')
    return {'path': str(path), 'bytes': s.st_size, 'allocated_bytes': s.st_blocks * 512,
            'inode': s.st_ino, 'device': s.st_dev, 'mtime_ns': s.st_mtime_ns, 'sha256': sha256(path)}


def inactive(root):
    state = json.loads((root / 'status.json').read_text())
    if (state.get('phase') not in ('gate_not_passed', 'complete', 'phase1_complete')
            or state.get('current_job') or state.get('controller_active') or living_processes(state)):
        raise RuntimeError(f'Experiment is not inactive: {root}')


def same_value(a, b):
    import math
    import torch
    if torch.is_tensor(a):
        return torch.is_tensor(b) and a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same_value(v, b[k]) for k, v in a.items())
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(same_value(x, y) for x, y in zip(a, b))
    if hasattr(a, '__dict__'):
        return hasattr(b, '__dict__') and same_value(vars(a), vars(b))
    if isinstance(a, float) and math.isnan(a):
        return isinstance(b, float) and math.isnan(b)
    return a == b


def compare_models(a, b):
    key = 'model_delta' if 'model_delta' in a else 'model'
    if (key not in b or a['epoch'] != b['epoch'] or not same_value(a[key], b[key])
            or any(a.get(k) != b.get(k) for k in ('format', 'base_checkpoint', 'base_sha256'))):
        raise ValueError('Retained checkpoint is not the same model')
    ignored = {'output_dir', 'resume', 'start_epoch', 'distributed', 'rank', 'gpu', 'world_size',
               'dist_url', 'dist_backend', 'num_workers', 'device', 'eval'}
    if 'args' in a and 'args' in b:
        aa = {k: v for k, v in vars(a['args']).items() if k not in ignored}
        bb = {k: v for k, v in vars(b['args']).items() if k not in ignored}
        if not same_value(aa, bb):
            raise ValueError('Retained model has a different inference recipe')
    return len(a[key])


def verify_initial_c1(checkpoint, base):
    import torch
    if checkpoint['epoch'] != -1:
        raise ValueError('Only the initial C1 fallback is redundant with C1')
    for name, tensor in checkpoint['model_delta'].items():
        if name.startswith(('cross_fusion_layers_vt.', 'cross_fusion_layers_tv.')):
            if not torch.equal(tensor, base['model'][name]):
                raise ValueError('Initial C1 projection changed')
        elif not name.startswith('region_anchor_exchange.'):
            raise ValueError('Unexpected tensor in the C1 fallback')
    if checkpoint['args'].visual_exchange != 'lavs':
        for branch in ('rgb_output', 'tir_output'):
            for field in ('weight', 'bias'):
                if checkpoint['model_delta'][f'region_anchor_exchange.{branch}.{field}'].count_nonzero():
                    raise ValueError('Anchor fallback is no longer equivalent to C1')


def audit(manifest):
    import torch
    torch.set_num_threads(4)
    if manifest.exists():
        raise FileExistsError('Cleanup audit already exists')
    result = {'user_authorized': True, 'created_at': time.time(), 'files': [], 'protected': {},
              'scope': 'verified duplicates and optimizer/RNG states from three completed rejected suites',
              'preserved': ['formal baselines and ablations', 'all best trained anchor checkpoints',
                            'inference weights for every completed last epoch', 'data, pretrained models, source and logs'],
              'finished': False}
    protected = set(STRUCTURE.glob('frozen/*.pth'))
    for root in SUITES:
        inactive(root)
        protected.update(root.glob('frozen/*.pth'))
        protected.update(root.glob('explore/*/trained_best_checkpoint.pth'))
    result['protected'] = {str(p): sha256(p) for p in sorted(protected)}
    write_json(manifest, result)
    for root in SUITES:
        for run in sorted((root / 'explore').iterdir()):
            if not run.is_dir():
                continue
            rows = [json.loads(s) for s in (run / 'log.txt').read_text().splitlines() if s.strip()]
            if [x['epoch'] for x in rows] != list(range(20)):
                raise ValueError(f'Incomplete run: {run}')
            best = run / 'best_checkpoint.pth'
            if best.exists():
                checkpoint = torch.load(best, map_location='cpu', weights_only=False)
                kept = root / 'frozen' / (run.name + '.pth')
                if root.name == '2026-10-26_region_anchor':
                    kept = Path(checkpoint['base_checkpoint'])
                    base = torch.load(kept, map_location='cpu', weights_only=False)
                    verify_initial_c1(checkpoint, base)
                    proof = 'base projections equal C1; anchor output projections are zero; epoch -1'
                    del base
                else:
                    retained = torch.load(kept, map_location='cpu', weights_only=False)
                    proof = {'equal_model_tensors': compare_models(checkpoint, retained)}
                    del retained
                result['files'].append({**info(best), 'kind': 'duplicate_best', 'root': str(root),
                                        'retained': str(kept), 'retained_sha256': sha256(kept), 'proof': proof, 'deleted': False})
                del checkpoint
            latest = run / 'checkpoint.pth'
            if latest.exists():
                checkpoint = torch.load(latest, map_location='cpu', weights_only=False)
                if checkpoint['epoch'] != 19 or 'model_delta' not in checkpoint or not RESUME_FIELDS.issubset(checkpoint):
                    raise ValueError('Latest checkpoint does not match the completed selective-tuning run')
                destination = root / 'compact_last' / (run.name + '_epoch19.pth')
                if destination.exists():
                    raise FileExistsError(destination)
                result['files'].append({**info(latest), 'kind': 'compact_completed_last', 'root': str(root),
                                        'destination': str(destination), 'epoch': 19, 'deleted': False})
                del checkpoint
            gc.collect()
            write_json(manifest, result)
    inactive(TEST_COPY)
    for variant in ('c1', 'joint'):
        path = TEST_COPY / variant / 'selected_model_only.pth'
        if not path.exists():
            continue
        kept = STRUCTURE / 'frozen' / f'{variant}_seed14.pth'
        a = torch.load(path, map_location='cpu', weights_only=False)
        b = torch.load(kept, map_location='cpu', weights_only=False)
        proof = compare_models(a, b)
        result['files'].append({**info(path), 'kind': 'duplicate_test_copy', 'root': str(TEST_COPY),
                                'retained': str(kept), 'retained_sha256': sha256(kept),
                                'proof': {'equal_model_tensors': proof}, 'deleted': False})
        del a, b
        gc.collect()
    result.update(audited_at=time.time(), planned_original_bytes=sum(x['bytes'] for x in result['files']))
    write_json(manifest, result)
    print(json.dumps({'audit': str(manifest), 'files': len(result['files']),
                      'original_GiB': result['planned_original_bytes'] / 1024**3}), flush=True)


def verify_protected(plan):
    for path, digest in plan['protected'].items():
        if sha256(path) != digest:
            raise ValueError(f'Retained checkpoint changed: {path}')


def execute(manifest):
    import torch
    torch.set_num_threads(4)
    plan = json.loads(manifest.read_text())
    if 'audited_at' not in plan:
        raise ValueError('Audit is incomplete')
    verify_protected(plan)
    allowed = {*SUITES, TEST_COPY}
    for item in plan['files']:
        if item['deleted']:
            continue
        path, root = Path(item['path']), Path(item['root'])
        if (root not in allowed or not path.is_relative_to(root)
                or path.name not in ('checkpoint.pth', 'best_checkpoint.pth', 'selected_model_only.pth')
                or str(path) in plan['protected']):
            raise ValueError('Deletion escaped the registered cleanup scope')
        inactive(root)
        current = info(path)
        if any(current[k] != item[k] for k in current):
            raise ValueError(f'Candidate changed since audit: {path}')
        if item['kind'] == 'compact_completed_last':
            destination = Path(item['destination'])
            if destination.parent != root / 'compact_last' or destination.exists():
                raise ValueError('Invalid compact checkpoint destination')
            old = torch.load(path, map_location='cpu', weights_only=False)
            compact = {k: v for k, v in old.items() if k not in RESUME_FIELDS}
            atomic_checkpoint(destination, compact)
            restored = torch.load(destination, map_location='cpu', weights_only=False)
            if not same_value(compact, restored):
                raise ValueError('Inference checkpoint conversion changed its payload')
            item['retained'] = str(destination)
            item['retained_sha256'] = sha256(destination)
            item['retained_bytes'] = destination.stat().st_size
            item['retained_allocated_bytes'] = destination.stat().st_blocks * 512
            item['proof'] = 'all non-resume fields, including every model tensor, round-trip equal'
            write_json(manifest, plan)
            del old, compact, restored
            gc.collect()
        elif sha256(item['retained']) != item['retained_sha256']:
            raise ValueError('Retained duplicate changed')
        inactive(root)
        current = info(path)
        if any(current[k] != item[k] for k in current):
            raise ValueError('Candidate changed before deletion')
        path.unlink()
        item.update(deleted=True, deleted_at=time.time())
        write_json(manifest, plan)
    verify_protected(plan)
    deleted = [x for x in plan['files'] if x['deleted']]
    plan.update(finished=True, finished_at=time.time(), deleted_files=len(deleted),
        removed_original_bytes=sum(x['bytes'] for x in deleted),
        newly_retained_bytes=sum(x.get('retained_bytes', 0) for x in deleted),
        released_bytes=sum(x['bytes'] - x.get('retained_bytes', 0) for x in deleted),
        released_allocated_bytes=sum(x['allocated_bytes'] - x.get('retained_allocated_bytes', 0) for x in deleted),
        protected_checksums_verified=True)
    write_json(manifest, plan)
    print(json.dumps({k: plan[k] for k in ('deleted_files', 'released_bytes', 'released_allocated_bytes',
                                          'protected_checksums_verified')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('audit', 'execute'))
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    (audit if args.action == 'audit' else execute)(args.manifest.resolve())
