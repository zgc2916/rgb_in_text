"""Audit and execute lossless compaction of completed experimental weights."""
import argparse
import gc
import json
from pathlib import Path
import time

from .cleanup_completed import info, inactive, same_value, compare_models, RESUME_FIELDS
from script_train.clip_lavs_control.runtime import ORIGINAL, sha256, write_json, atomic_checkpoint
from utils.checkpoint_storage import pack_relative, unpack_relative

BASE = ORIGINAL / 'output_clip_structure_control/2026-10-05_c1_detail_boundary/frozen/c1_seed13.pth'
RECENT = tuple(ORIGINAL / 'output_clip_lavs_control' / name for name in (
    '2026-10-27_native_reobserve', '2026-10-27_spatial_quality'))
OLDER = tuple(ORIGINAL / 'output_clip_lavs_control' / name for name in (
    '2026-10-09_text_local_exchange', '2026-10-13_soft_spatial_exchange'))
DEFAULT = ORIGINAL / 'output_checkpoint_cleanup/2026-10-28_lossless_compaction/cleanup_manifest.json'


def audit(manifest):
    import torch
    torch.set_num_threads(4)
    if manifest.exists():
        raise FileExistsError(manifest)
    protected = set((BASE.parent).glob('*.pth'))
    for root in RECENT:
        inactive(root)
        protected.update(root.glob('explore/*/best_checkpoint.pth'))
    plan = {'created_at': time.time(), 'user_authorized': True, 'files': [], 'finished': False,
            'protected': {str(p): sha256(p) for p in sorted(protected)},
            'preserved': ['formal C1 and ablations', 'all distinct best and last inference models',
                          'all data, pretrained resources, recipes, source and metrics']}
    base = torch.load(BASE, map_location='cpu', weights_only=False)
    for root in RECENT:
        for run in sorted((root / 'explore').iterdir()):
            if not run.is_dir():
                continue
            rows = [json.loads(s) for s in (run / 'log.txt').read_text().splitlines() if s.strip()]
            if [x['epoch'] for x in rows] != list(range(20)):
                raise ValueError('A recent run is incomplete')
            best, duplicate = run / 'best_checkpoint.pth', run / 'trained_best_checkpoint.pth'
            if duplicate.exists():
                a = torch.load(best, map_location='cpu', weights_only=False)
                b = torch.load(duplicate, map_location='cpu', weights_only=False)
                compare_models(a, b)
                if not same_value(a, b):
                    raise ValueError('Duplicate checkpoint inference metadata differs')
                plan['files'].append({**info(duplicate), 'root': str(root), 'kind': 'duplicate',
                    'retained': str(best), 'retained_sha256': sha256(best), 'deleted': False})
                del a, b
            latest = run / 'checkpoint.pth'
            if latest.exists():
                state = torch.load(latest, map_location='cpu', weights_only=False)
                if state['epoch'] != 19 or not RESUME_FIELDS.issubset(state):
                    raise ValueError('Last checkpoint is not a completed resumable state')
                plan['files'].append({**info(latest), 'root': str(root), 'kind': 'resume_state',
                    'destination': str(root / 'compact_last' / (run.name + '_epoch19.pth')), 'deleted': False})
                del state
            gc.collect()
    for root in OLDER:
        inactive(root)
        for path in sorted((root / 'frozen').glob('*.pth')):
            state = torch.load(path, map_location='cpu', weights_only=False)
            compact = pack_relative(state, base['model'], BASE, plan['protected'][str(BASE)])
            if not same_value(state, unpack_relative(compact, base['model'])):
                raise ValueError('Relative model reconstruction changed tensors or metadata')
            plan['files'].append({**info(path), 'root': str(root), 'kind': 'relative_model',
                'destination': str(root / 'compact_frozen' / path.name),
                'model_tensors': len(state['model']), 'changed_tensors': len(compact['model_delta']),
                'deleted': False})
            del state, compact
            gc.collect()
    plan.update(audited_at=time.time(), original_bytes=sum(x['bytes'] for x in plan['files']))
    write_json(manifest, plan)
    print(json.dumps({'files': len(plan['files']), 'original_GiB': plan['original_bytes']/1024**3}), flush=True)


def execute(manifest):
    import torch
    torch.set_num_threads(4)
    plan = json.loads(manifest.read_text())
    if 'audited_at' not in plan:
        raise ValueError('Audit is incomplete')
    for path, digest in plan['protected'].items():
        if sha256(path) != digest:
            raise ValueError('A protected model changed after audit')
    base = torch.load(BASE, map_location='cpu', weights_only=False)
    for item in plan['files']:
        if item['deleted']:
            continue
        path, root = Path(item['path']), Path(item['root'])
        if root not in (*RECENT, *OLDER) or not path.is_relative_to(root) or str(path) in plan['protected']:
            raise ValueError('Cleanup escaped the registered workspace scope')
        inactive(root)
        if info(path) != {k: item[k] for k in info(path)}:
            raise ValueError('Cleanup candidate changed after audit')
        if item['kind'] != 'duplicate':
            destination = Path(item['destination'])
            if (not destination.is_relative_to(root) or destination.is_symlink()
                    or destination.parent.name not in ('compact_last', 'compact_frozen')):
                raise ValueError('Unexpected compact model destination')
            old = torch.load(path, map_location='cpu', weights_only=False)
            compact = ({k: v for k, v in old.items() if k not in RESUME_FIELDS}
                       if item['kind'] == 'resume_state' else
                       pack_relative(old, base['model'], BASE, plan['protected'][str(BASE)]))
            if destination.exists():
                if not same_value(compact, torch.load(destination, map_location='cpu', weights_only=False)):
                    raise ValueError('An existing destination differs from the planned conversion')
            else:
                atomic_checkpoint(destination, compact)
            restored = torch.load(destination, map_location='cpu', weights_only=False)
            if item['kind'] == 'relative_model':
                restored = unpack_relative(restored, base['model'])
                if not same_value(old, restored):
                    raise ValueError('Relative model round trip failed')
            elif not same_value(compact, restored):
                raise ValueError('Last inference state round trip failed')
            item.update(retained=str(destination), retained_sha256=sha256(destination),
                        newly_retained_bytes=destination.stat().st_size,
                        round_trip_all_tensors_and_metadata_equal=True)
            write_json(manifest, plan)
            del old, compact, restored
            gc.collect()
        elif sha256(item['retained']) != item['retained_sha256']:
            raise ValueError('The retained duplicate changed')
        inactive(root)
        current = info(path)
        if any(current[k] != item[k] for k in current):
            raise ValueError('Candidate changed before deletion')
        path.unlink()
        item.update(deleted=True, deleted_at=time.time())
        write_json(manifest, plan)
    for path, digest in plan['protected'].items():
        if sha256(path) != digest:
            raise ValueError('A protected checkpoint changed during cleanup')
    relocations = {x['path']: {'retained': x['retained'], 'sha256': x['retained_sha256'],
                   'storage': x['kind']} for x in plan['files']}
    write_json(manifest.parent / 'checkpoint_relocations.json', relocations)
    for root in (*RECENT, *OLDER):
        write_json(root / 'cleanup_notice_2026-10-28.json', {'manifest': str(manifest),
            'relocations': {k: v for k, v in relocations.items() if Path(k).is_relative_to(root)},
            'completed_resume_state_removed': root in RECENT,
            'relative_model_loader': 'utils.checkpoint_storage.load_checkpoint'})
    plan.update(finished=True, finished_at=time.time(), protected_checksums_verified=True,
        released_bytes=sum(x['bytes']-x.get('newly_retained_bytes',0) for x in plan['files']))
    write_json(manifest, plan)
    print(json.dumps({'deleted_original_files':len(plan['files']), 'released_GiB':plan['released_bytes']/1024**3}),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('audit','execute'))
    parser.add_argument('--manifest',type=Path,default=DEFAULT)
    args=parser.parse_args()
    (audit if args.action=='audit' else execute)(args.manifest.resolve())
