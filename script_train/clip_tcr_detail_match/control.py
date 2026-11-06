"""Detached 120-epoch TCR detail and boundary ablations on RefM3FD."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from . import runtime as r
from script_train.clip_cross_dataset.control import verify_snapshot,gpu_pair,object_hash
from script_train.clip_soft_control.control import process_identity,living_processes,environment

MODULE='script_train.clip_tcr_detail_match'
DEFAULT_ROOT=r.ORIGINAL/'output_clip_lavs_control/2026-11-05_m3fd_tcr_detail'


def prepare(root):
    import torch
    if root.exists():raise FileExistsError(root)
    names=[f'{d}_{m}_seed{r.SEED}' for d in r.DATASETS for m in r.METHODS]
    recipes={f'{d}_{m}_seed{r.SEED}':vars(r.recipe(root/'runs'/f'{d}_{m}_seed{r.SEED}',d,m))
             for d in r.DATASETS for m in r.METHODS}
    config=next(iter(recipes.values()))
    if r.sha256(r.BASELINE)!=r.BASELINE_SHA256:raise ValueError('Reference TCR checkpoint changed')
    resources=[Path(config['retrain']),r.BASELINE,r.BASELINE_ROOT/'report.json']
    resources += [p for p in Path(config['pretrained_model_path']).rglob('*') if p.is_file()]
    resources += [p for p in Path(config['dataset_bert_model_path']).rglob('*') if p.is_file() and p.stat().st_size<10*1024**2]
    datasets={}
    for ds in r.DATASETS.values():
        paths={s:Path(config['split_root'])/ds/f'{ds}_{s}.pth' for s in ('train','val','test')}
        resources.extend(paths.values())
        rows={s:torch.load(p,map_location='cpu',weights_only=False) for s,p in paths.items()}
        files={s:set(str(x[0]) for x in items) for s,items in rows.items()}
        overlap={f'{a}_{b}':sorted(files[a]&files[b]) for a,b in (('train','val'),('train','test'),('val','test'))}
        if overlap['train_test'] or overlap['val_test']:raise ValueError('Test filename overlap')
        # Match DistributedSampler + drop_last batch sampler + accumulation final group.
        per_rank=(len(rows['train'])+1)//2
        microbatches=per_rank//config['batch_size']
        datasets[ds]={'counts':{s:len(x) for s,x in rows.items()},'filename_overlap':overlap,
                      'microbatches_per_rank_per_epoch':microbatches,
                      'optimizer_updates_per_epoch':(microbatches+config['grad_accum_steps']-1)//config['grad_accum_steps']}
    for p in resources:
        if not p.is_file():raise FileNotFoundError(p)
    root.mkdir(parents=True);source=root/'source';source.mkdir()
    for name in ('models','datasets','utils','train_val','script_train','script_eval','tests'):
        shutil.copytree(r.ORIGINAL/name,source/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc','.pytest_cache'))
    shutil.copy2(r.ORIGINAL/'engine.py',source/'engine.py')
    r.write_json(source/'run_resources/c1_recipe.json',json.loads((r.REFERENCE/'formal/c1_seed13/recipe.json').read_text()))
    r.write_json(root/'source_manifest.json',{str(p.relative_to(source)):r.sha256(p) for p in sorted(source.rglob('*')) if p.is_file()})
    r.write_json(root/'resource_manifest.json',{str(p.resolve()):r.sha256(p) for p in resources})
    prov={'source_sha256':r.sha256(root/'source_manifest.json'),'resources_sha256':r.sha256(root/'resource_manifest.json'),
          'recipes_sha256':object_hash(recipes),'python':r.PYTHON}
    manifest={'version':1,'provenance':prov,'recipes':recipes,'runs':names,'datasets':datasets,
              'epochs_per_run':r.EPOCHS,'seed':r.SEED,'initialization':{'kind':'public MMVG mixup checkpoint',
              'checkpoint':config['retrain'],'sha256':r.sha256(config['retrain']),
              'uses_ref_flir_finetuned_weights':False},
              'method':'TCR-query with detail, boundary, and combined; compare with the previously trained TCR-query.',
              'baseline':{'checkpoint':str(r.BASELINE),'sha256':r.BASELINE_SHA256,'report':str(r.BASELINE_ROOT/'report.json'),'dataset':'rgbtvg_m3fd'},
              'matching':'Same 120 epochs, dataset, split, batch, optimizer, lr, HiLoRA schedule, C1 and TCR losses and seed; ablated additions use their registered auxiliary losses.',
              'selection':'first strict best trained-epoch validation Acc@0.5; all three sealed before test',
              'test_used_for_selection':False,'test_previously_observed':True,
              'split_caveat':'One train/val image filename overlap in RefM3FD; no same-name test overlap; content overlap unverified.',
              'resume_caveat':'Native C1 trainer restores optimizer and scheduler, but not per-rank RNG states.'}
    r.write_json(root/'preregistration.json',manifest)
    r.write_json(root/'status.json',{'phase':'prepared','jobs':{},'training_started':False,'completed_epochs':0,
                                     'controller_active':False,'test_evaluated':False})


def launch(root,gpus,recover):
    verify_snapshot(root);state=json.loads((root/'status.json').read_text())
    if living_processes(state):raise RuntimeError('Existing live process')
    if state.get('phase')=='complete':raise RuntimeError('Already complete')
    if state.get('training_started') and not recover:raise RuntimeError('Use resume')
    gpus=gpu_pair(state.get('gpus',gpus) if recover else gpus)
    command=['nohup',r.PYTHON,'-u','-m',MODULE+'.control','controller','--root',str(root),'--gpus',gpus]
    if recover:command.append('--recover')
    env=environment(gpus,root);env['PYTHONPATH']=str(root/'source')
    with (root/'controller.log').open('a') as log:
        proc=subprocess.Popen(command,cwd=root/'source',env=env,stdin=subprocess.DEVNULL,
                              stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    r.write_json(root/'launch.json',{'pid':proc.pid,'argv':command,'gpus':gpus,'method':'nohup and independent session'})
    for _ in range(30):
        time.sleep(1);state=json.loads((root/'status.json').read_text())
        if state.get('phase')=='failed':raise RuntimeError(state.get('error'))
        if living_processes(state):print(json.dumps(state,indent=2));return
    raise RuntimeError('Detached controller failed to start')


def controller(root,gpus,recover):
    import torch
    lock=(root/'suite.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest=verify_snapshot(root);state=json.loads((root/'status.json').read_text())
    if living_processes(state):raise RuntimeError('Previous job still alive')
    state.update(controller_pid=os.getpid(),controller_identity=process_identity(os.getpid()),controller_active=True,
                 phase='gpu_preflight',gpus=gpus,error=None)
    def heartbeat():
        progress={}
        for name in manifest['runs']:
            p=root/'runs'/name/'log.txt';rows=[]
            if p.exists():
                for line in p.read_text().splitlines():
                    if line.startswith('{'):
                        try:rows.append(json.loads(line))
                        except json.JSONDecodeError:pass
            progress[name]={'completed_epochs':len(rows),'last_epoch':rows[-1]['epoch'] if rows else None,
                            'validation_acc05':rows[-1].get('validation_accu') if rows else None}
        state['runs']=progress;state['completed_epochs']=sum(v['completed_epochs'] for v in progress.values())
        state['updated_at']=time.time();r.write_json(root/'status.json',state)
    def execute(name,command,log):
        if state['jobs'].get(name,{}).get('completed'):return
        start=time.monotonic()
        while True:
            try:gpu_pair(gpus);break
            except RuntimeError as error:
                state.update(phase='waiting_for_gpus',current_job=name,waiting_reason=str(error));heartbeat()
                if time.monotonic()-start>=3600:raise
                time.sleep(10)
        state.update(phase='gpu_preflight' if name=='preflight' else 'testing' if name.startswith('evaluate_') else 'training',
                     current_job=name,waiting_reason=None)
        log.parent.mkdir(parents=True,exist_ok=True)
        r.write_json(log.with_suffix('.command.json'),{'argv':command,'gpus':gpus})
        env=environment(gpus,root);env['PYTHONPATH']=str(root/'source')
        with log.open('a') as stream:
            proc=subprocess.Popen(command,cwd=root/'source',env=env,stdin=subprocess.DEVNULL,stdout=stream,
                                  stderr=subprocess.STDOUT,start_new_session=True)
            state.update(job_pid=proc.pid,job_identity=process_identity(proc.pid))
            if name!='preflight':state['training_started']=True
            heartbeat()
            while proc.poll() is None:
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:heartbeat()
        state['jobs'][name]={'completed':proc.returncode==0,'exit_code':proc.returncode}
        state.update(job_pid=None,job_identity=None);heartbeat()
        if proc.returncode:raise RuntimeError(f'{name} failed ({proc.returncode}), see {log}')
    torchrun=[r.PYTHON,'-m','torch.distributed.run','--standalone','--nproc_per_node=2','-m']
    heartbeat()
    try:
        execute('preflight',torchrun+[MODULE+'.preflight','--root',str(root)],root/'preflight/console.log')
        if not json.loads((root/'preflight.json').read_text())['passed']:raise RuntimeError('Preflight failed')
        state['gpu_acceptance_passed']=True
        for name in manifest['runs']:
            ds='m3fd';method=name.removeprefix('m3fd_').removesuffix('_seed13')
            if method not in r.METHODS:raise ValueError('Unexpected registered method')
            cmd=torchrun+[MODULE+'.control','train','--root',str(root),'--dataset',ds,'--method',method]
            if recover and (root/'runs'/name/'checkpoint.pth').exists():cmd.append('--recover')
            execute(name,cmd,root/'runs'/name/'console.log')
        selections={}
        for name in manifest['runs']:
            directory=root/'runs'/name
            rows=[json.loads(line) for line in (directory/'log.txt').read_text().splitlines() if line.startswith('{')]
            if len(rows)!=r.EPOCHS or [v['epoch'] for v in rows]!=list(range(r.EPOCHS)):raise ValueError('Incomplete training log')
            original=directory/'best_checkpoint.pth';data=torch.load(original,map_location='cpu',weights_only=False)
            best=max(rows,key=lambda v:v['validation_accu'])
            if data['epoch']!=best['epoch'] or abs(data['val_accu']-best['validation_accu'])>1e-12:
                raise ValueError('Best checkpoint is not validation-selected')
            digest=r.sha256(original);frozen=root/'frozen'/f'{name}.pth';frozen.parent.mkdir(exist_ok=True)
            if not frozen.exists():shutil.copy2(original,frozen)
            if r.sha256(frozen)!=digest:raise ValueError('Frozen checkpoint differs')
            selections[name]={'checkpoint':str(frozen),'original_checkpoint':str(original),'sha256':digest,
                              'epoch':data['epoch'],'weight_view':'raw','validation':data['val_stats']}
            del data
        registration=root/'test_registration.json'
        if registration.exists():
            old=json.loads(registration.read_text())
            if json.dumps(old['selected'],sort_keys=True)!=json.dumps(selections,sort_keys=True):raise ValueError('Test registration changed')
        else:r.write_json(registration,{'selected':selections,'selection_used_test':False,'test_previously_observed':True})
        for name,item in selections.items():
            out=root/'evaluation'/name
            execute('evaluate_'+name,torchrun+[MODULE+'.evaluate','--checkpoint',item['checkpoint'],
                    '--output-root',str(out),'--splits','val','test'],out/'console.log')
        from script_eval.clip_differential_test import verify_result
        evaluation={name:{s:json.loads((root/'evaluation'/name/f'{s}.json').read_text()) for s in ('val','test')}
                    for name in selections}
        verification={}
        for name,item in selections.items():
            ds=manifest['recipes'][name]['dataset']
            for split in ('val','test'):
                verification[f'{name}/{split}']=verify_result(evaluation[name][split],item,split,manifest['datasets'][ds]['counts'][split])
            if any(r.sha256(item[k])!=item['sha256'] for k in ('checkpoint','original_checkpoint')):
                raise ValueError('Checkpoint changed during evaluation')
        r.write_json(root/'verification.json',{'passed':True,'results':verification})
        baseline_report=json.loads((r.BASELINE_ROOT/'report.json').read_text())
        if r.sha256(r.BASELINE)!=r.BASELINE_SHA256:raise ValueError('Reference checkpoint changed')
        report={'stage':'complete','baseline_reference':{'checkpoint':str(r.BASELINE),'sha256':r.BASELINE_SHA256,
                'validation':baseline_report['validation']['m3fd_tcr_query_seed13'],
                'test':baseline_report['test']['m3fd_tcr_query_seed13']},'validation':{n:v['val']['summary'] for n,v in evaluation.items()},
                'test':{n:v['test']['summary'] for n,v in evaluation.items()},
                'test_groups':{n:v['test']['groups'] for n,v in evaluation.items()},
                'single_model_inference':True,'test_used_for_selection':False,'test_previously_observed':True}
        r.write_json(root/'report.json',report);state.update(phase='complete',current_job=None,test_evaluated=True,test=report['test'])
    except BaseException as error:
        state.update(phase='failed',error=str(error));raise
    finally:state['controller_active']=False;heartbeat()


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','launch','resume','controller','train','status'))
    p.add_argument('--root',type=Path,default=DEFAULT_ROOT);p.add_argument('--gpus',default='auto')
    p.add_argument('--dataset',choices=r.DATASETS);p.add_argument('--method',choices=r.METHODS)
    p.add_argument('--recover',action='store_true');a=p.parse_args();root=a.root.resolve()
    if a.action=='prepare':prepare(root);print(root)
    elif a.action in ('launch','resume'):launch(root,a.gpus,a.action=='resume')
    elif a.action=='controller':controller(root,a.gpus,a.recover)
    elif a.action=='train':r.train(root,a.dataset,a.method,a.recover)
    else:print(json.dumps(json.loads((root/'status.json').read_text()),indent=2,ensure_ascii=False))

if __name__=='__main__':main()
