"""Detached, bounded warm-start suite with checkpoint recovery."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import threading
import time

from .runtime import (REPOSITORY, REFERENCE, PYTHON, VARIANTS, ORDER, EPOCHS,
    recipe, references, sha256, write_json, best_from_log, choose_promotion)
from script_train.clip_soft_control.control import process_identity, living_processes, environment, gpu_check

DEFAULT_ROOT = REPOSITORY/'output_clip_lavs_control/2026-10-18_clean_joint_control'
MODULE = 'script_train.clip_training_control'
TERMINAL = ('gate_not_passed','complete_promoted','complete_no_stable_gain')


def recipes_hash(recipes):
    return hashlib.sha256(json.dumps(recipes,sort_keys=True).encode()).hexdigest()


def snapshot(root,gpus):
    if root.exists():
        raise FileExistsError('Experiment already exists; use resume')
    root.mkdir(parents=True)
    destination=root/'source'
    destination.mkdir()
    for name in ('models','datasets','utils','train_val','script_train','script_eval','tests'):
        shutil.copytree(REPOSITORY/name,destination/name,
            ignore=shutil.ignore_patterns('__pycache__','*.pyc','.pytest_cache'))
    shutil.copy2(REPOSITORY/'engine.py',destination/'engine.py')
    write_json(destination/'run_resources/c1_recipe.json',json.loads((REFERENCE/'formal/c1_seed13/recipe.json').read_text()))
    selected=references()
    write_json(destination/'run_resources/c1_references.json',selected)
    historical=REPOSITORY/'output_clip_lavs_control/2026-10-18_reliable_exchange_warm/explore/direction_gate_seed13'
    for name in ('recipe.json','log.txt'):
        shutil.copy2(historical/name,destination/'run_resources'/('augmented_direction_'+name))
    files={str(p.relative_to(destination)):sha256(p) for p in sorted(destination.rglob('*')) if p.is_file()}
    write_json(root/'source_manifest.json',files)
    config=recipe(root)
    resources={}
    for path in sorted(Path(config.pretrained_model_path).rglob('*')):
        if path.is_file():
            resources[str(path.resolve())]=sha256(path)
    for split in ('train','val'):
        path=Path(config.split_root)/config.dataset/f'{config.dataset}_{split}.pth'
        resources[str(path.resolve())]=sha256(path)
        out=root/'resources/annotations'/path.name
        out.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,out)
    for path in sorted(Path(config.dataset_bert_model_path).rglob('*')):
        if path.is_file() and path.stat().st_size<10*1024**2:
            resources[str(path.resolve())]=sha256(path)
    for item in selected:
        path=REFERENCE/'evaluation'/f"c1_seed{item['seed']}"/'val.json'
        resources[str(path.resolve())]=sha256(path)
    write_json(root/'resource_manifest.json',resources)
    recipes={f'{v}_seed{s}':vars(recipe(root/'explore'/f'{v}_seed{s}',v,s))
             for v in VARIANTS for s in (13,14,15)}
    source_hash=sha256(root/'source_manifest.json')
    provenance={'source_manifest_sha256':source_hash,'resource_manifest_sha256':sha256(root/'resource_manifest.json'),
                'recipes_sha256':recipes_hash(recipes),'gpus':gpus,'python':PYTHON,
                'base_checkpoints':{str(item['seed']):item['sha256'] for item in selected}}
    write_json(root/'preregistration.json',{'version':1,'stage':'trained-C1 controlled continuation',
        'provenance':provenance,'source_manifest_sha256':source_hash,'references':selected,
        'gpus':gpus,'python':PYTHON,'socket':'clip_clean_'+root.name.replace('-','_'),'session':'suite',
        'recipes':recipes,'phase1':[f'{v}_seed13' for v in ORDER],
        'epochs_per_run':EPOCHS,'maximum_new_runs':8,'maximum_automatic_retries_per_job':1,
        'selection':'first strict maximum validation Acc@0.5 INCLUDING initial C1 as epoch -1',
        'controls':{
            'a_clean_full':'direction gate, no synthetic corruption, original losses, original weights frozen',
            'b_clean_box':'A with only L1/GIoU grounding loss and unchanged gate penalty',
            'c_joint_box':'B with existing external Q/K/V/O projections trainable at lr1e-5',
            'd_c1_continue':'C1 with the same projection scope, data, grounding loss and continuation budget as C'},
        'historical_control':'A vs previous 20-epoch direction gate with synthetic corruption; recipe and log copied into source/run_resources',
        'phase1_gate':{'joint_vs_original_minimum_extra_hits':4,'joint_vs_continued_minimum_extra_hits':1,
            'control_vs_original_minimum_extra_hits':4,'quality_non_decrease':['accu_07','miou','miou_weak']},
        'continuation':'If C passes both baselines, repeat C and D on seeds14/15; otherwise repeat D alone only if it improves C1',
        'replication_gate':{'performance_mean_acc05_delta_minimum':.005,'fusion_mean_paired_acc05_delta_minimum':.003,
            'positive_seeds_minimum':2,'negative_seeds_maximum':0,
            'quality_non_decreasing_means':['accu_07','miou','miou_weak']},
        'test_selection':False,'automatic_test_evaluation':False,'test_previously_observed':True,
        'method':{'initialization':'seed-matched trained C1 validation-selected checkpoint',
            'corruption_probability':0.,'geometry_augmentation':'unchanged C1 target-safe augmentation',
            'effective_batch':96,'new_lr':1e-4,'existing_projection_lr':1e-5,'weight_decay':1e-4,
            'scheduler':'2-epoch warmup times cosine20 with minimum factor0.1; same as previous screen',
            'frozen':['CLIP and both LoRA adapters','internal language bridges','IAFv3','six VL layers','bbox head'],
            'training_behavior':'frozen modules stay eval; gradients flow to explicitly enabled parameters',
            'added_parameters_abc':1741360,'added_parameters_d':0,'quality_auxiliary_loss':False,
            'ground_truth_used_in_forward':False,'formal_common_init_comparison_completed':False,
            'initialization_acceptance':{'full_model_max_box_error':1e-5,
                'validation_acc05_acc07_hits':'must match C1 exactly','miou_numerical_tolerance':1e-5}}})


def verify_snapshot(root):
    manifest=json.loads((root/'preregistration.json').read_text())
    if sha256(root/'source_manifest.json')!=manifest['source_manifest_sha256']:
        raise ValueError('Source manifest changed')
    for name,expected in json.loads((root/'source_manifest.json').read_text()).items():
        if sha256(root/'source'/name)!=expected:
            raise ValueError(f'Frozen source changed: {name}')
    provenance=manifest['provenance']
    if recipes_hash(manifest['recipes'])!=provenance['recipes_sha256']:
        raise ValueError('Recipes changed')
    if manifest['gpus']!=provenance['gpus'] or manifest['python']!=provenance['python']:
        raise ValueError('Runtime configuration changed')
    if sha256(root/'resource_manifest.json')!=provenance['resource_manifest_sha256']:
        raise ValueError('Resource manifest changed')
    for path,expected in json.loads((root/'resource_manifest.json').read_text()).items():
        if sha256(path)!=expected:
            raise ValueError(f'Runtime resource changed: {path}')
    for item in manifest['references']:
        if item['sha256']!=provenance['base_checkpoints'][str(item['seed'])] or sha256(item['checkpoint'])!=item['sha256']:
            raise ValueError('C1 reference checkpoint changed')
    return manifest


def launch(root,gpus,resume=False):
    if resume:
        manifest=verify_snapshot(root)
        state=json.loads((root/'status.json').read_text()) if (root/'status.json').exists() else {}
        live=living_processes(state)
        if live:
            raise RuntimeError(f'Controller/training already alive: {live}')
        if state.get('phase') in TERMINAL:
            raise RuntimeError('Suite already completed')
        if gpus!=manifest['gpus']:
            raise ValueError('Recovery must use registered GPU configuration')
    else:
        snapshot(root,gpus)
        manifest=verify_snapshot(root)
    if subprocess.run(['tmux','-L',manifest['socket'],'has-session','-t','suite'],
                      stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0:
        raise RuntimeError('tmux session already exists; refusing duplicate launch')
    command=[PYTHON,'-u','-m',MODULE+'.control','controller','--root',str(root),'--gpus',gpus]
    if resume:
        command.append('--recover')
    shell=(f'cd {shlex.quote(str(root/"source"))} && {shlex.join(command)} '
           f'>> {shlex.quote(str(root/"controller.log"))} 2>&1 < /dev/null; '
           'clip_clean_rc=$?; exit "$clip_clean_rc"')
    subprocess.run(['tmux','-L',manifest['socket'],'-f','/dev/null','new-session','-d','-s','suite',shell],
        check=True,env=environment(gpus,root),stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,start_new_session=True)
    for _ in range(30):
        time.sleep(1)
        if (root/'status.json').exists():
            state=json.loads((root/'status.json').read_text())
            if state.get('phase')=='failed':
                raise RuntimeError(state.get('error'))
            if any(name=='controller' for name,_ in living_processes(state)):
                print(json.dumps({'launched':True,'root':str(root),'status':state},indent=2),flush=True)
                return
    raise RuntimeError('No healthy controller status after 30 seconds; inspect controller.log')


def freeze(root,variant,seed):
    from .runtime import atomic_checkpoint
    import torch
    name=f'{variant}_seed{seed}'
    directory=root/'explore'/name
    best=best_from_log(directory)
    out=root/'frozen'/f'{name}.pth'
    if not out.exists():
        checkpoint=torch.load(directory/'best_checkpoint.pth',map_location='cpu',weights_only=False)
        if checkpoint['epoch']!=best['epoch']:
            raise AssertionError('Best metadata/checkpoint epoch mismatch')
        atomic_checkpoint(out,checkpoint)
    checkpoint=torch.load(out,map_location='cpu',weights_only=False)
    if checkpoint['epoch']!=best['epoch'] or checkpoint['val_stats']['accu']!=best['accu']:
        raise AssertionError('Frozen model differs from selection')
    item={'name':name,'variant':variant,'seed':seed,'epoch':best['epoch'],'checkpoint':str(out),
          'sha256':sha256(out),'base_checkpoint':checkpoint['base_checkpoint'],'base_sha256':checkpoint['base_sha256'],
          'validation':best}
    previous=root/'selected_models.json'
    selected=json.loads(previous.read_text()) if previous.exists() else []
    existing=next((old for old in selected if old['name']==name),None)
    if existing and existing!=item:
        raise ValueError('Frozen checkpoint registration changed')
    if not existing:
        selected.append(item)
        write_json(previous,selected)
    return item


def controller(root,gpus,recover=False):
    lock=(root/'suite.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    manifest=json.loads((root/'preregistration.json').read_text())
    state=json.loads((root/'status.json').read_text()) if recover and (root/'status.json').exists() else {'jobs':{}}
    if state.get('job_pid') and process_identity(state['job_pid'])==state.get('job_identity'):
        raise RuntimeError('Previous job is still alive; refusing duplicate recovery')
    state.update(controller_pid=os.getpid(),controller_identity=process_identity(os.getpid()),
        socket=manifest['socket'],session='suite',phase='preflight',source_snapshot=str(root/'source'),
        started_at=time.time(),error=None,gpus=gpus,epochs_per_run=EPOCHS,
        maximum_new_runs=manifest['maximum_new_runs'],protocol='clean-data A/B/C/D controls; fixed validation selection')
    heartbeat_lock=threading.RLock()
    stopped=threading.Event()
    def heartbeat():
        with heartbeat_lock:
            job=state.get('current_job','') or ''
            if job.startswith('explore/'):
                progress=root/job/'progress.json'
                if progress.exists():
                    state.update(json.loads(progress.read_text()))
            completed=[name for name,entry in state['jobs'].items() if name.startswith('explore/') and entry.get('completed')]
            state['completed_training_runs']=len(completed)
            partial=(state.get('completed_epochs',0) or 0) if job.startswith('explore/') and job not in completed else 0
            state['total_completed_epochs']=len(completed)*EPOCHS+partial
            state['heartbeat_at']=time.time()
            write_json(root/'status.json',state)
    def pulse():
        while not stopped.wait(20):
            heartbeat()
    heartbeat()
    threading.Thread(target=pulse,daemon=True).start()
    torchrun=[PYTHON,'-u','-m','torch.distributed.run','--standalone','--nproc_per_node=2','-m']
    def execute(name,command,log):
        if state['jobs'].get(name,{}).get('completed'):
            return
        previous_phase=state['phase']
        maximum=2 if name.startswith('explore/') else 1
        for attempt in range(maximum):
            while True:
                try:
                    gpu_check(gpus)
                    break
                except RuntimeError as error:
                    state.update(phase='waiting_for_gpus',current_job=name,waiting_reason=str(error))
                    heartbeat()
                    time.sleep(20)
            state.update(phase=previous_phase,current_job=name,epoch=None,completed_epochs=0,
                epoch_in_progress=False,best=None,exit_code=None,waiting_reason=None,attempt=attempt+1)
            heartbeat()
            log.parent.mkdir(parents=True,exist_ok=True)
            argv=list(command)
            if name.startswith('explore/') and (root/name/'checkpoint.pth').exists() and '--recover' not in argv:
                argv.append('--recover')
            write_json(log.with_suffix('.command.json'),{'argv':argv,'gpus':gpus,'cwd':str(root/'source')})
            print(f'Start {name}, attempt {attempt+1}: {log}',flush=True)
            with log.open('a') as stream:
                process=subprocess.Popen(argv,cwd=root/'source',env=environment(gpus,root),stdin=subprocess.DEVNULL,
                    stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                state.update(job_pid=process.pid,job_identity=process_identity(process.pid))
                heartbeat()
                while process.poll() is None:
                    try:
                        process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        heartbeat()
            state['jobs'][name]={'completed':process.returncode==0,'exit_code':process.returncode,
                                'finished_at':time.time(),'attempt':attempt+1}
            state['exit_code']=process.returncode
            heartbeat()
            if process.returncode==0:
                print(f'Completed {name}',flush=True)
                return
            print(f'Failed {name}, exit {process.returncode}; retry allowed={attempt+1<maximum}',flush=True)
        raise RuntimeError(f'{name} failed; inspect {log}')
    def run_one(variant,seed):
        name=f'{variant}_seed{seed}'
        directory=root/'explore'/name
        execute('explore/'+name,[*torchrun,MODULE+'.control','train','--root',str(root),
            '--run-dir',str(directory),'--variant',variant,'--seed',str(seed)],directory/'console.log')
        item=freeze(root,variant,seed)
        out=root/'evaluation'/name
        if not (out/'val.json').exists():
            execute('validation/'+name,[*torchrun,MODULE+'.evaluate','--checkpoint',item['checkpoint'],
                '--output-root',str(out),'--splits','val'],out/'console.log')
    try:
        manifest=verify_snapshot(root)
        execute('preflight',[*torchrun,MODULE+'.control','preflight','--root',str(root)],root/'preflight/console.log')
        if not json.loads((root/'preflight.json').read_text())['passed']:
            raise AssertionError('Preflight did not pass')
        state['phase']='training_controls'
        heartbeat()
        for variant in ORDER:
            run_one(variant,13)
        runs={v:best_from_log(root/'explore'/f'{v}_seed13') for v in VARIANTS}
        baseline=next(item['validation'] for item in manifest['references'] if item['seed']==13)
        promotion=choose_promotion(baseline,runs)
        write_json(root/'promotion.json',promotion)
        selected=promotion['selected_variant']
        if selected:
            state['phase']='seed_replication'
            heartbeat()
            for seed in (14,15):
                for variant in promotion['replicate_variants']:
                    run_one(variant,seed)
        from .report import report
        result=report(root)
        write_json(root/'report.json',result)
        state.update(phase=('complete_promoted' if result['replication']['passed'] else
                           'complete_no_stable_gain') if selected else 'gate_not_passed',
                     current_job=None,finished_at=time.time(),recommended_model=result['recommended_model'])
        heartbeat()
    except BaseException as error:
        state.update(phase='failed',error=str(error),failed_at=time.time())
        heartbeat()
        raise
    finally:
        stopped.set()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('launch','resume','status','attach','controller','train','preflight'))
    parser.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    parser.add_argument('--gpus')
    parser.add_argument('--recover',action='store_true')
    parser.add_argument('--run-dir',type=Path)
    parser.add_argument('--variant',choices=VARIANTS,default='a_clean_full')
    parser.add_argument('--seed',type=int,default=13)
    cli=parser.parse_args()
    root=cli.root.resolve()
    if cli.gpus is None:
        cli.gpus=json.loads((root/'preregistration.json').read_text())['gpus'] if cli.action=='resume' else '2,3'
    if cli.action in ('launch','resume'):
        launch(root,cli.gpus,cli.action=='resume')
    elif cli.action=='status':
        state=json.loads((root/'status.json').read_text())
        state['live_processes']=living_processes(state)
        print(json.dumps(state,indent=2,ensure_ascii=False))
    elif cli.action=='attach':
        manifest=json.loads((root/'preregistration.json').read_text())
        os.execvp('tmux',['tmux','-L',manifest['socket'],'attach-session','-t','suite'])
    elif cli.action=='controller':
        controller(root,cli.gpus,cli.recover)
    elif cli.action=='train':
        from .runtime import train
        train(root,cli.run_dir.resolve(),cli.variant,cli.seed,cli.recover)
    else:
        from .preflight import run
        run(root)


if __name__=='__main__':
    main()
