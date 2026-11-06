"""Compact finished experiment recovery checkpoints after explicit user request."""
import gc
import json
import time
from pathlib import Path

import torch

from script_train.clip_lavs_control.runtime import ORIGINAL, atomic_checkpoint, sha256, write_json
from script_train.clip_training_control.cleanup_completed import info, inactive, same_value, RESUME_FIELDS

SPECS = {
    '2026-11-01_differential_lavs': 60,
    '2026-11-01_gentle_differential': 60,
    '2026-11-01_lavs_replacement': 40,
    '2026-10-31_dense_grid': 20,
    '2026-10-28_late_adaptation_ema_r1': 30,
}
ROOT = ORIGINAL / 'output_clip_lavs_control'
MANIFEST = ORIGINAL / 'output_checkpoint_cleanup/2026-11-02_completed_training/cleanup_manifest.json'


def run():
    torch.set_num_threads(4)
    if MANIFEST.exists():
        raise FileExistsError('Cleanup already started; inspect the manifest before repeating')
    protected = set((ORIGINAL / 'output_clip_structure_control/2026-10-05_c1_detail_boundary/frozen').glob('*.pth'))
    paths = []
    active_resources = set()
    for status in ROOT.glob('*/status.json'):
        state = json.loads(status.read_text())
        if state.get('controller_active') or state.get('current_job'):
            resources = status.parent / 'resource_manifest.json'
            if resources.exists():
                active_resources.update(Path(x).resolve() for x in json.loads(resources.read_text()))
    for name, epochs in SPECS.items():
        root = ROOT / name
        inactive(root)
        state = json.loads((root / 'status.json').read_text())
        for path in sorted(root.glob('explore/*/checkpoint.pth')):
            progress = json.loads((path.parent / 'progress.json').read_text())
            job = state['jobs'].get(path.parent.name, {})
            if progress.get('completed_epochs') != epochs or not job.get('completed') or job.get('exit_code') != 0:
                raise ValueError(f'Run is not complete: {path}')
            if path.resolve() in active_resources:
                raise ValueError(f'Current training depends on this checkpoint: {path}')
            best = path.parent / 'best_checkpoint.pth'
            if not best.is_file():
                raise FileNotFoundError(best)
            protected.update(path.parent.glob('*best_checkpoint.pth'))
            paths.append((root, path, epochs))
    protected_hashes = {str(p): sha256(p) for p in sorted(protected)}
    plan = {'user_authorized': True, 'created_at': time.time(), 'finished': False,
            'scope': 'Completed training only; preserve last raw model and EMA, remove optimizer/scheduler/RNG',
            'exact_resume_available_for_compacted_runs': False,
            'protected_sha256': protected_hashes, 'files': []}
    for root, path, epochs in paths:
        state = torch.load(path, map_location='cpu', weights_only=False)
        if (state.get('epoch') != epochs - 1 or 'model_delta' not in state
                or not RESUME_FIELDS.issubset(state) or 'ema_state' not in state):
            raise ValueError(f'Unexpected final state: {path}')
        dest = root / 'compact_last' / f'{path.parent.name}_epoch{epochs-1}.pth'
        if dest.exists():
            raise FileExistsError(dest)
        plan['files'].append({**info(path), 'root': str(root), 'destination': str(dest),
                              'epoch': epochs-1, 'deleted': False})
        del state
        gc.collect()
    plan['audited_at'] = time.time()
    write_json(MANIFEST, plan)
    print(f"Audited {len(paths)} completed checkpoints", flush=True)
    for item in plan['files']:
        path, root, dest = Path(item['path']), Path(item['root']), Path(item['destination'])
        inactive(root)
        current = info(path)
        if any(current[k] != item[k] for k in current):
            raise ValueError(f'Checkpoint changed after audit: {path}')
        old = torch.load(path, map_location='cpu', weights_only=False)
        compact = {k:v for k,v in old.items() if k not in RESUME_FIELDS}
        atomic_checkpoint(dest, compact)
        restored = torch.load(dest, map_location='cpu', weights_only=False)
        if not same_value(compact, restored):
            raise ValueError(f'Compact checkpoint did not round-trip exactly: {path}')
        item.update(retained=str(dest), retained_sha256=sha256(dest),
                    retained_bytes=dest.stat().st_size, all_retained_fields_equal=True,
                    removed_fields=sorted(RESUME_FIELDS))
        write_json(MANIFEST, plan)
        del old, compact, restored
        gc.collect()
        inactive(root)
        current = info(path)
        if any(current[k] != item[k] for k in current):
            raise ValueError(f'Checkpoint changed before deletion: {path}')
        path.unlink()
        item.update(deleted=True, deleted_at=time.time())
        write_json(MANIFEST, plan)
        print(f"Compacted {path.parent.name}: {item['bytes']-item['retained_bytes']} bytes saved", flush=True)
    for path, digest in protected_hashes.items():
        if sha256(path) != digest:
            raise ValueError(f'Protected checkpoint changed: {path}')
    relocation = {x['path']: {'retained':x['retained'], 'sha256':x['retained_sha256']} for x in plan['files']}
    write_json(MANIFEST.parent / 'checkpoint_relocations.json', relocation)
    for name in SPECS:
        root = ROOT / name
        write_json(root / 'cleanup_notice_2026-11-02.json', {'manifest':str(MANIFEST),
            'relocations':{k:v for k,v in relocation.items() if Path(k).is_relative_to(root)},
            'best_and_trained_best_preserved':True, 'last_raw_model_and_ema_preserved':True,
            'exact_training_resume_available':False})
    plan.update(finished=True, finished_at=time.time(), protected_checksums_verified=True,
                released_bytes=sum(x['bytes']-x['retained_bytes'] for x in plan['files']))
    write_json(MANIFEST, plan)
    print(f"Finished: {plan['released_bytes']/1024**3:.2f} GiB reclaimed", flush=True)


if __name__ == '__main__':
    run()
