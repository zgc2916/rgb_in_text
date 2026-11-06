"""Diagnostic evaluation explicitly requested after the dense-grid screen.

Run directly. Training and evaluation code are imported from the immutable
training snapshot. All three validation selections are sealed before test.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)


def phase_one_ready(root, names, epochs):
    """Wait for durable final commits, rather than the last log append alone."""
    for name in names:
        path = root/'explore'/name/'progress.json'
        if not path.exists():
            return False
        progress = json.loads(path.read_text())
        if progress.get('completed_epochs') != epochs or progress.get('epoch_in_progress', True):
            return False
    return True


def aligned_samples(cases, total):
    """Cross-resolution comparisons share expressions, not canvas coordinates."""
    reference = None
    for name, rows in cases.items():
        if len(rows) != total or [row['index'] for row in rows] != list(range(total)):
            raise ValueError(f'Incomplete or unordered evaluation: {name}')
        identities = [(row['index'], row['image'], row['phrase']) for row in rows]
        if reference is None:
            reference = identities
        elif identities != reference:
            raise ValueError('Evaluation expressions differ')


def summarize(rows):
    if not rows:
        return {'total': 0}
    return {'total': len(rows), 'images': len({x['image'] for x in rows}),
            'correct_05': sum(x['iou'] >= .5 for x in rows),
            'correct_07': sum(x['iou'] >= .7 for x in rows),
            'acc_05': sum(x['iou'] >= .5 for x in rows)/len(rows),
            'acc_07': sum(x['iou'] >= .7 for x in rows)/len(rows),
            'miou': sum(x['iou'] for x in rows)/len(rows)}


def paired_changes(rows, reference):
    aligned_samples({'candidate': rows, 'reference': reference}, len(reference))
    return {'gained_hits05': sum(a['iou'] >= .5 > b['iou'] for a,b in zip(rows,reference)),
            'lost_hits05': sum(b['iou'] >= .5 > a['iou'] for a,b in zip(rows,reference)),
            'net_hits05': sum(a['iou'] >= .5 for a in rows)-sum(b['iou'] >= .5 for b in reference),
            'miou_delta_percentage_points': sum(a['iou']-b['iou'] for a,b in zip(rows,reference))/len(rows)*100,
            'coordinate_note': 'Each model evaluated against its own resolution-specific padded GT; identity paired by expression.'}


def build_report(root, output, plan):
    import torch
    cases = list(plan['checkpoints'])
    evaluations = {name: {split: json.loads((output/name/f'{split}.json').read_text())
                          for split in ('val','test')} for name in cases}
    for split, total in [('val',608), ('test',2104)]:
        aligned_samples({name: data[split]['samples'] for name,data in evaluations.items()}, total)
        for name,data in evaluations.items():
            source=data[split]['source']
            selected=plan['checkpoints'][name]
            if (source['sha256'] != selected['sha256'] or source['epoch'] != selected['epoch']
                    or source['weight_view'] != selected['weight_view']):
                raise ValueError('Result checkpoint does not match sealed plan')
            if sha256(plan['checkpoints'][name]['checkpoint']) != plan['checkpoints'][name]['sha256']:
                raise ValueError('Frozen checkpoint changed during evaluation')
            if split=='val':
                for current,key in [('acc_05','accu'),('acc_07','accu_07'),('miou','miou')]:
                    tolerance=1e-5 if key=='miou' else 1e-12
                    if abs(data[split]['summary'][current]-selected['validation'][key])>tolerance:
                        raise ValueError('Evaluation did not reproduce registered validation metrics')
    annotations = torch.load(plan['annotations']['test']['path'], map_location='cpu', weights_only=False)
    results = {}
    for name,data in evaluations.items():
        rows = data['test']['samples']
        results[name] = {'val': summarize(data['val']['samples']), 'test': summarize(rows),
            'groups': {'small': summarize([row for row,ann in zip(rows,annotations) if str(ann[5]).lower()=='small']),
                       'normal': summarize([row for row,ann in zip(rows,annotations) if str(ann[5]).lower()!='small']),
                       'weak': summarize([row for row,ann in zip(rows,annotations) if str(ann[4]) in ('1','2')])}}
        for key in ('correct_05','correct_07','acc_05','acc_07','miou'):
            if abs(results[name]['test'][key]-data['test']['summary'][key]) > 1e-12:
                raise ValueError('Saved summary and sample metrics differ')
    reference = evaluations['c1_224_seed13']['test']['samples']
    report = {'kind':'user_requested_diagnostic_test','training_seed':13,'multi_seed_replication':False,
        'training_root':str(root),'evaluation_plan_sha256':sha256(output/'evaluation_plan.json'),
        'test_used_for_selection':False,'original_training_gate_unchanged':True,
        'runs':results,'paired_vs_matched_224':{
            name:paired_changes(data['test']['samples'],reference) for name,data in evaluations.items()
            if name!='c1_224_seed13'},
        'target_acc05':.74,'target_reached':{name:result['test']['acc_05']>=.74 for name,result in results.items()}}
    training_report = root/'report.json'
    if training_report.exists():
        report['training_promotion'] = json.loads(training_report.read_text()).get('promotion')
    write_json(output/'report.json',report)
    lines=['# Dense-grid validation-selected diagnostic test','',
           'All checkpoints fixed by validation before test; seed13 only.','',
           '| Model | Test Acc05 | Test Acc07 | Test mIoU | Hits05 / 2104 |',
           '| --- | ---: | ---: | ---: | ---: |']
    for name,result in results.items():
        s=result['test']
        lines.append(f"| {name} | {s['acc_05']*100:.3f}% | {s['acc_07']*100:.3f}% | {s['miou']*100:.3f}% | {s['correct_05']} |")
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    return report


def run(root, output, gpus, wait_seconds):
    output.mkdir(parents=True,exist_ok=True)
    lock = (output/'evaluation.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    sys.path.insert(0,str(root/'source'))
    from script_train.clip_dense_grid_control import runtime as r
    from script_train.clip_dense_grid_control.control import verify_snapshot, gpu_pair
    names = [f'{v}_seed13' for v in r.VARIANTS]
    def status(**values):
        write_json(output/'status.json',{'pid':os.getpid(),'updated_at':time.time(),**values})
    try:
        deadline = time.monotonic()+wait_seconds
        while not phase_one_ready(root,names,r.EPOCHS):
            training = json.loads((root/'status.json').read_text())
            if training.get('phase')=='failed':
                raise RuntimeError('Training failed: '+str(training.get('error')))
            if time.monotonic()>=deadline:
                raise TimeoutError('Training did not finish before evaluation wait deadline')
            status(phase='waiting_for_training',completed_training_epochs=training.get('completed_epochs'),
                   current_training_job=training.get('current_job'),completed=[],error=None)
            time.sleep(10)
        registration = verify_snapshot(root)
        selected = r.collect_best_runs(root)
        plan_path = output/'evaluation_plan.json'
        if plan_path.exists():
            plan = json.loads(plan_path.read_text())
        else:
            cases = {}
            for name in names:
                original = root/'explore'/name/'best_checkpoint.pth'
                digest = sha256(original)
                frozen = output/'frozen'/f'{name}.pth'
                frozen.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(original,frozen)
                if sha256(frozen)!=digest or sha256(original)!=digest:
                    raise ValueError('Selected weights changed while freezing')
                cases[name]={'checkpoint':str(frozen),'original_checkpoint':str(original),
                    'sha256':digest,'epoch':selected[name]['epoch'],'weight_view':selected[name]['weight_view'],
                    'validation':selected[name],'resolution':registration['recipes'][name]['imsize']}
            a = r.recipe(output,'c1_224')
            annotations={split:Path(a.split_root)/a.dataset/f'{a.dataset}_{split}.pth' for split in ('val','test')}
            plan={'training_root':str(root),'source_manifest_sha256':registration['provenance']['source_sha256'],
                'runner_sha256':sha256(Path(__file__)),'created_at':time.time(),'checkpoints':cases,
                'annotations':{split:{'path':str(p),'sha256':sha256(p)} for split,p in annotations.items()},
                'user_requested_diagnostic':True,'validation_gate_not_overridden_for_training':True,
                'selection':'First strict best validation Acc05 after all 20 epochs, including initial; no test selection',
                'test_used_for_selection':False,'protocol':{'world_size':2,'batch_size_per_gpu':12,
                    'ground_truth_in_forward':False,'grad_context':'torch.no_grad','optimizer_created':False,
                    'single_model_inference':True,'splits':['val','test']}}
            write_json(plan_path,plan)
        if plan['runner_sha256']!=sha256(Path(__file__)):
            raise ValueError('Diagnostic runner changed after sealing selections')
        if plan['source_manifest_sha256']!=sha256(root/'source_manifest.json'):
            raise ValueError('Training source manifest changed')
        for item in plan['annotations'].values():
            if sha256(item['path'])!=item['sha256']:
                raise ValueError('Evaluation annotation changed')
        completed=[]
        for name in names:
            item=plan['checkpoints'][name]
            if sha256(item['checkpoint'])!=item['sha256']:
                raise ValueError('Frozen diagnostic checkpoint changed')
            gpu_deadline=time.monotonic()+600
            while True:
                try:
                    gpu_pair(gpus)
                    break
                except RuntimeError as error:
                    if time.monotonic()>=gpu_deadline:
                        raise
                    status(phase='waiting_for_gpus',current_case=name,completed=completed,
                           waiting_reason=str(error),error=None)
                    time.sleep(10)
            destination=output/name
            destination.mkdir(exist_ok=True)
            command=[r.PYTHON,'-m','torch.distributed.run','--standalone','--nproc_per_node=2','-m',
                     'script_train.clip_dense_grid_control.evaluate','--checkpoint',item['checkpoint'],
                     '--output-root',str(destination),'--splits','val','test']
            write_json(destination/'command.json',{'argv':command,'cwd':str(root/'source'),'gpus':gpus})
            environment=os.environ.copy()
            environment.update(CUDA_VISIBLE_DEVICES=gpus,PYTHONPATH=str(root/'source'),
                OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
                TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1',MPLCONFIGDIR=str(output/'matplotlib'))
            with (destination/'console.log').open('a') as stream:
                process=subprocess.Popen(command,cwd=root/'source',env=environment,
                    stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                while process.poll() is None:
                    status(phase='evaluating',current_case=name,job_pid=process.pid,completed=completed,error=None)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        pass
            if process.returncode:
                raise RuntimeError(f'{name} evaluation failed; see {destination}/console.log')
            completed.append(name)
        report=build_report(root,output,plan)
        status(phase='complete',completed=completed,error=None,test_used_for_selection=False)
        print(json.dumps({name:result['test'] for name,result in report['runs'].items()},indent=2),flush=True)
    except BaseException as error:
        status(phase='failed',error=str(error))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpus',default='0,2')
    parser.add_argument('--wait-seconds',type=int,default=3600)
    cli=parser.parse_args()
    run(cli.root.resolve(),cli.output.resolve(),cli.gpus,cli.wait_seconds)
