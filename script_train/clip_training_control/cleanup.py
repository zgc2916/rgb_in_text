"""Audit and remove only proven redundant/completed-run checkpoint files."""
import argparse
import gc
import json
from pathlib import Path
import stat
import time

from script_train.clip_soft_control.runtime import sha256, write_json, ORIGINAL
from script_train.clip_soft_control.control import living_processes

ROOTS = (
    ORIGINAL/'output_clip_structure_control/2026-10-05_c1_detail_boundary',
    ORIGINAL/'output_clip_lavs_control/2026-10-09_text_local_exchange',
    ORIGINAL/'output_clip_lavs_control/2026-10-13_soft_spatial_exchange',
)
AUDIT = ORIGINAL/'output_checkpoint_cleanup/2026-10-18_redundant_formal/cleanup_manifest.json'


def inactive(root):
    state=json.loads((root/'status.json').read_text())
    if state.get('phase') not in ('complete','gate_not_passed','phase1_complete') or state.get('current_job'):
        raise RuntimeError(f'Experiment is not completed: {root}')
    if living_processes(state):
        raise RuntimeError(f'Experiment still has a live process: {root}')
    return state['phase']


def file_info(path):
    path=Path(path)
    if path.is_symlink() or not path.resolve().is_relative_to(ORIGINAL):
        raise ValueError(f'Unsafe cleanup path: {path}')
    s=path.stat()
    if not stat.S_ISREG(s.st_mode) or s.st_nlink!=1:
        raise ValueError(f'Not an unlinked regular checkpoint: {path}')
    return {'path':str(path),'bytes':s.st_size,'allocated_bytes':s.st_blocks*512,
            'inode':s.st_ino,'device':s.st_dev,'mtime_ns':s.st_mtime_ns,'sha256':sha256(path)}


def audit(destination):
    import torch
    torch.set_num_threads(4)
    if destination.exists():
        raise FileExistsError('Audit already exists; inspect it or execute the registered plan')
    result={'user_authorized':True,'scope':'redundant formal best copies and latest states from completed rejected suites',
        'created_at':time.time(),'preserved':['all frozen checkpoints','all datasets and pretrained weights',
            'all recipes, source snapshots, logs and evaluations','all new/active experiment checkpoints',
            'learned last states from the recent compact warm-start experiments'],
        'files':[],'skipped':[],'finished':False}
    write_json(destination,result)
    for root in ROOTS:
        phase=inactive(root)
        selected=json.loads((root/'selected_models.json').read_text())
        for item in selected:
            name=f"{item['variant']}_seed{item['seed']}"
            kept=Path(item['checkpoint'])
            if not kept.resolve().is_relative_to(root/'frozen') or sha256(kept)!=item['sha256']:
                raise ValueError('Frozen retained checkpoint failed identity check')
            best=root/'formal'/name/'best_checkpoint.pth'
            if not best.exists():
                result['skipped'].append({'path':str(best),'reason':'already absent'})
                continue
            first=torch.load(best,map_location='cpu',weights_only=False)
            second=torch.load(kept,map_location='cpu',weights_only=False)
            same=(first['epoch']==second['epoch']==item['epoch'] and first['model'].keys()==second['model'].keys()
                  and all(torch.equal(first['model'][key],second['model'][key]) for key in first['model']))
            if not same:
                raise ValueError(f'Best checkpoint is not a duplicate of frozen model: {best}')
            tensor_count=len(first['model'])
            del first,second
            gc.collect()
            result['files'].append({**file_info(best),'kind':'redundant_best','experiment_root':str(root),
                'completed_phase':phase,'retained_checkpoint':str(kept),'retained_sha256':item['sha256'],
                'model_tensors_verified_equal':tensor_count,'epoch':item['epoch'],'deleted':False})
            write_json(destination,result)
            latest=root/'formal'/name/'checkpoint.pth'
            # The structure suite retains best optimizer checkpoints only;
            # never classify them as disposable latest training states.
            if root==ROOTS[0] or not latest.exists():
                continue
            rows=[json.loads(line) for line in (latest.parent/'log.txt').read_text().splitlines() if line.strip()]
            if [row['epoch'] for row in rows]!=list(range(120)):
                raise ValueError(f'Latest checkpoint belongs to incomplete training: {latest}')
            result['files'].append({**file_info(latest),'kind':'completed_rejected_latest',
                'experiment_root':str(root),'completed_phase':phase,'completed_epochs':120,
                'retained_checkpoint':str(kept),'retained_sha256':item['sha256'],'deleted':False})
            write_json(destination,result)
    result['planned_bytes']=sum(item['bytes'] for item in result['files'])
    result['planned_allocated_bytes']=sum(item['allocated_bytes'] for item in result['files'])
    result['audited_at']=time.time()
    write_json(destination,result)
    print(json.dumps({'audit':str(destination),'files':len(result['files']),
                      'bytes':result['planned_bytes'],'GiB':result['planned_bytes']/1024**3}),flush=True)


def execute(destination):
    result=json.loads(destination.read_text())
    if 'audited_at' not in result:
        raise ValueError('Audit is not complete')
    allowed={root.resolve() for root in ROOTS}
    for root in ROOTS:
        inactive(root)
    for item in result['files']:
        if item['deleted']:
            continue
        root=Path(item['experiment_root'])
        path=Path(item['path'])
        if root.resolve() not in allowed or path.parent.parent!=root/'formal':
            raise ValueError('Cleanup path escaped the registered formal directory')
        if path.name not in ('best_checkpoint.pth','checkpoint.pth'):
            raise ValueError('Unexpected checkpoint filename')
        kept=Path(item['retained_checkpoint'])
        if sha256(kept)!=item['retained_sha256']:
            raise ValueError('Retained frozen model changed before deletion')
        actual=file_info(path)
        if any(actual[key]!=item[key] for key in actual):
            raise ValueError('Candidate changed since audit')
        inactive(root)
        path.unlink()
        item.update(deleted=True,deleted_at=time.time())
        write_json(destination,result)
    result.update(finished=True,finished_at=time.time(),deleted_files=sum(item['deleted'] for item in result['files']),
        deleted_bytes=sum(item['bytes'] for item in result['files'] if item['deleted']),
        deleted_allocated_bytes=sum(item['allocated_bytes'] for item in result['files'] if item['deleted']))
    write_json(destination,result)
    print(json.dumps({'deleted_files':result['deleted_files'],'deleted_bytes':result['deleted_bytes'],
                      'GiB':result['deleted_bytes']/1024**3,'manifest':str(destination)}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('audit','execute'))
    parser.add_argument('--manifest',type=Path,default=AUDIT)
    args=parser.parse_args()
    if args.action=='audit':
        audit(args.manifest.resolve())
    else:
        execute(args.manifest.resolve())


if __name__=='__main__':
    main()
