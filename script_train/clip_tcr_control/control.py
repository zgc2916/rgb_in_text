"""Detached matched controls for target-conditioned LAVS attention."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time

from . import runtime as r
from script_train.clip_soft_control.control import process_identity, living_processes, environment

MODULE = 'script_train.clip_tcr_control'
DEFAULT_ROOT = r.REPOSITORY / 'output_clip_lavs_control/2026-11-02_tcr_target'
TERMINAL = ('complete', 'gate_not_passed', 'complete_no_stable_gain')


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def prepare(root, cpu_report=None):
    if root.exists():
        raise FileExistsError('Experiment exists; use resume with its frozen source')
    config = r.recipe(root)
    resources = [Path(r.reference(seed)['checkpoint']) for seed in r.SEEDS]
    resources += [p for p in Path(config.pretrained_model_path).rglob('*') if p.is_file()]
    resources += [p for p in Path(config.dataset_bert_model_path).rglob('*') if p.is_file() and p.stat().st_size < 10*1024**2]
    resources += [Path(config.split_root)/config.dataset/f'{config.dataset}_{split}.pth' for split in ('train','val','test')]
    resources += [r.REFERENCE/f'evaluation/c1_seed{seed}/val.json' for seed in r.SEEDS]
    audit = r.REPOSITORY/'output_clip_analysis/2026-11-01_lavs_attention_audit/original_c1_batch_audit.json'
    if audit.is_file():
        resources.append(audit)
    mask_audit = r.REPOSITORY/'output_clip_analysis/2026-11-01_differential_preparation/initial_mask_ablation.json'
    if mask_audit.is_file():
        resources.append(mask_audit)
    for path in resources:
        if not path.is_file():
            raise FileNotFoundError(path)
    if not Path(config.pretrained_model_path,'pytorch_model.bin').is_file():
        raise FileNotFoundError('Local CLIP weights required')
    root.mkdir(parents=True)
    source = root/'source'
    source.mkdir()
    for name in ('models','datasets','utils','train_val','script_train','script_eval','tests'):
        shutil.copytree(r.REPOSITORY/name,source/name,
                        ignore=shutil.ignore_patterns('__pycache__','*.pyc','.pytest_cache'))
    shutil.copy2(r.REPOSITORY/'engine.py',source/'engine.py')
    r.write_json(source/'run_resources/c1_recipe.json',json.loads((r.REFERENCE/'formal/c1_seed13/recipe.json').read_text()))
    r.write_json(source/'run_resources/c1_references.json',r.references())
    files={str(p.relative_to(source)):r.sha256(p) for p in sorted(source.rglob('*')) if p.is_file()}
    r.write_json(root/'source_manifest.json',files)
    r.write_json(root/'resource_manifest.json',{str(p.resolve()):r.sha256(p) for p in resources})
    recipes={f'{v}_seed{seed}':vars(r.recipe(root/'explore'/f'{v}_seed{seed}',v,seed))
             for seed in r.SEEDS for v in r.VARIANTS}
    provenance={'source_sha256':r.sha256(root/'source_manifest.json'),
                'resources_sha256':r.sha256(root/'resource_manifest.json'),
                'recipes_sha256':object_hash(recipes),'python':r.PYTHON}
    manifest={'version':1,'provenance':provenance,'recipes':recipes,
        'references':[r.reference(seed) for seed in r.SEEDS],
        'socket':'clip_tcr_target_'+root.name.replace('-','_'),
        'phase1':[f'{v}_seed13' for v in r.ORDER],
        'replication':[f'{v}_seed{seed}' for seed in (14,15) for v in r.VARIANTS],
        'epochs_per_run':r.EPOCHS,'maximum_training_runs':len(r.VARIANTS)*len(r.SEEDS),
        'maximum_epochs':len(r.VARIANTS)*len(r.SEEDS)*r.EPOCHS,
        'experiment_scope':'C1 warm-start structure screen; public-initialization full retraining is a subsequent experiment',
        'selection':'first strict best validation Acc05 including the structure-specific initial; raw wins same-epoch EMA ties',
        'initialization':'paired original C1 seeds, not test-selected seed or latest adaptation',
        'method':{'resolutions':[224,224,224],'visual_grids':[14,14,14],
                  'student':'paired C1 warm-start, batch-corrected migrated LAVS with differential branch disabled',
                  'controls':['no target supervision','target supervision without attention bias',
                              'target supervision with target key bias'],
                  'target_usage':'GT boxes enter training loss only, never model forward',
                  'inference_models':1,'extra_inference_views':0,'effective_batch':96,
                  'microbatch_per_gpu':4,'accumulation':12,'vision_lora_lr':4e-6,
                  'inherited_and_migrated_lr':2e-6,'tcr_lr':5e-5,'target_loss_weight':0.05,
                  'teacher_training_only':False,'data_pipeline':'original C1 shared by all controls',
                  'test_driven_selection':False,
                  'novelty':'Text-conditioned grounding and target attention have prior work; a task-specific claim requires ablations and external validation'},
        'test_used_for_selection':False,'test_previously_observed':True,
        'test_protocol':'User-requested diagnostic after all registered training; validation gate still determines promotion; all cases sealed before test',
        'target':{'metric':'test Acc@0.5','threshold':.74,'required_hits_of_2104':1557}}
    if cpu_report is not None:
        data=json.loads(cpu_report.read_text())
        if not data['passed']:
            raise ValueError('CPU acceptance failed')
        for name,digest in data['source_sha256'].items():
            if r.sha256(r.REPOSITORY/name)!=digest:
                raise ValueError('Source changed after CPU acceptance')
        shutil.copy2(cpu_report,root/'cpu_acceptance.json')
        manifest['cpu_acceptance_sha256']=r.sha256(root/'cpu_acceptance.json')
    r.write_json(root/'preregistration.json',manifest)
    r.write_json(root/'status.json',{'phase':'prepared','training_started':False,'completed_epochs':0,
        'controller_active':False,'source_snapshot':str(source),'jobs':{},'test_evaluated':False})
    return manifest


def verify_snapshot(root):
    manifest = json.loads((root / 'preregistration.json').read_text())
    p = manifest['provenance']
    if r.sha256(root / 'source_manifest.json') != p['source_sha256']:
        raise ValueError('Source manifest changed')
    for name, digest in json.loads((root / 'source_manifest.json').read_text()).items():
        if r.sha256(root / 'source' / name) != digest:
            raise ValueError(f'Frozen source changed: {name}')
    if r.sha256(root / 'resource_manifest.json') != p['resources_sha256']:
        raise ValueError('Resource manifest changed')
    for path, digest in json.loads((root / 'resource_manifest.json').read_text()).items():
        if r.sha256(path) != digest:
            raise ValueError(f'Runtime resource changed: {path}')
    if object_hash(manifest['recipes']) != p['recipes_sha256'] or p['python'] != r.PYTHON:
        raise ValueError('Registered recipe or Python changed')
    if 'cpu_acceptance_sha256' in manifest and r.sha256(root / 'cpu_acceptance.json') != manifest['cpu_acceptance_sha256']:
        raise ValueError('CPU acceptance report changed')
    return manifest


def gpu_pair(requested='auto'):
    probe = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.free',
                            '--format=csv,noheader,nounits'], capture_output=True, text=True)
    if probe.returncode:
        raise RuntimeError('NVIDIA driver unavailable: ' + (probe.stderr or probe.stdout).strip())
    available = {int(row.split(',')[0]): tuple(int(x) for x in row.split(',')[1:])
                 for row in probe.stdout.splitlines() if row.strip()}
    selected = ([i for i, (used, free) in sorted(available.items()) if used <= 2048 and free >= 18000][:2]
                if requested == 'auto' else [int(i) for i in requested.split(',')])
    if len(selected) != 2 or len(set(selected)) != 2:
        raise RuntimeError('Two distinct free GPUs with at least 18000 MB each are required')
    for index in selected:
        used, free = available.get(index, (float('inf'), 0))
        if used > 2048 or free < 18000:
            raise RuntimeError(f'GPU {index} unavailable: used={used}MB, free={free}MB')
    return ','.join(map(str, selected))


def freeze_test_selections(root, names, runs, selected):
    """Seal every validation selection before any diagnostic test starts."""
    selections={}
    for name in names:
        source=root/'explore'/name/'best_checkpoint.pth'
        digest=r.sha256(source)
        frozen=root/'frozen'/f'{name}.pth'
        frozen.parent.mkdir(parents=True,exist_ok=True)
        if frozen.exists():
            if r.sha256(frozen)!=digest:
                raise ValueError('Existing frozen checkpoint differs from validation selection')
        else:
            temporary=frozen.with_suffix('.pth.tmp')
            shutil.copy2(source,temporary)
            if r.sha256(temporary)!=digest or r.sha256(source)!=digest:
                raise ValueError('Checkpoint changed during test freeze')
            temporary.replace(frozen)
        selections[name]={'checkpoint':str(frozen),'original_checkpoint':str(source),'sha256':digest,
                          'epoch':runs[name]['epoch'],'weight_view':runs[name]['weight_view']}
    registration=root/'test_registration.json'
    if registration.exists():
        if json.loads(registration.read_text())['selected']!=selections:
            raise ValueError('Sealed test selection changed')
    else:
        r.write_json(registration,{'selected':selections,'selection_used_test':False,
            'selected_variant':selected,'created_at':time.time(),
            'kind':'user_requested_diagnostic_after_training',
            'test_previously_observed':True,'validation_gate_unchanged':True})
    return selections


def launch(root, requested='auto', resume=False):
    manifest = verify_snapshot(root)
    state = json.loads((root / 'status.json').read_text())
    if living_processes(state):
        raise RuntimeError('A controller or training job already owns this experiment')
    if state['phase'] in TERMINAL:
        raise RuntimeError('Screening already finished')
    if state.get('training_started') and not resume:
        raise RuntimeError('Training has already started; use resume to recover optimizer and RNG')
    try:
        gpus = gpu_pair(state.get('gpus', requested) if resume else requested)
    except (OSError, RuntimeError) as error:
        state.update(phase='waiting_for_gpu', training_started=bool(state.get('training_started', False)),
                     controller_active=False, waiting_reason=str(error), updated_at=time.time())
        r.write_json(root / 'status.json', state)
        print(json.dumps(state, indent=2, ensure_ascii=False))
        return False
    command = ['nohup', r.PYTHON, '-u', '-m', MODULE + '.control', 'controller', '--root', str(root), '--gpus', gpus]
    if resume:
        command.append('--recover')
    env = environment(gpus, root)
    env['PYTHONPATH'] = str(root/'source')
    with (root/'controller.log').open('a') as stream:
        process = subprocess.Popen(command, cwd=root/'source', env=env,
            stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    r.write_json(root/'launch.json', {'argv':command,'pid':process.pid,'method':'nohup plus new OS session',
                                    'created_at':time.time(),'gpus':gpus})
    for _ in range(30):
        time.sleep(1)
        current = json.loads((root / 'status.json').read_text())
        if current.get('phase') == 'failed':
            raise RuntimeError(current['error'])
        if living_processes(current):
            print(json.dumps(current, indent=2))
            return True
    raise RuntimeError('Detached controller failed to become healthy; inspect controller.log')


def controller(root, gpus, recover=False):
    lock = (root / 'suite.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = verify_snapshot(root)
    state = json.loads((root / 'status.json').read_text())
    if living_processes(state):
        raise RuntimeError('Previous controller or torchrun is still alive')
    state.update(controller_pid=os.getpid(), controller_identity=process_identity(os.getpid()),
                 controller_active=True, gpus=gpus, phase='gpu_preflight', error=None,
                 source_snapshot=str(root / 'source'))
    def heartbeat():
        progress_files = list((root / 'explore').glob('*/progress.json'))
        state['completed_epochs'] = sum(json.loads(path.read_text()).get('completed_epochs', 0)
                                        for path in progress_files)
        job = state.get('current_job', '')
        if job in manifest['recipes']:
            path = root / 'explore' / job / 'progress.json'
            if path.exists():
                state['current_progress'] = json.loads(path.read_text())
        state['updated_at'] = time.time()
        r.write_json(root / 'status.json', state)
    def execute(name, command, log):
        if state['jobs'].get(name, {}).get('completed'):
            return
        wait_start = time.monotonic()
        while True:
            try:
                gpu_pair(gpus)
                break
            except RuntimeError as error:
                state.update(phase='waiting_for_gpus', waiting_reason=str(error), current_job=name)
                heartbeat()
                if time.monotonic() - wait_start >= 3600:
                    raise RuntimeError('GPU availability did not recover within one hour') from error
                time.sleep(10)
        state.update(phase=('gpu_preflight' if name == 'preflight' else
                            'final_test_evaluation' if name.startswith('evaluate_') else 'training_controls'),
                     current_job=name, waiting_reason=None)
        log.parent.mkdir(parents=True, exist_ok=True)
        r.write_json(log.with_suffix('.command.json'), {'argv': command, 'gpus': gpus, 'cwd': str(root / 'source')})
        with log.open('a') as stream:
            process = subprocess.Popen(command, cwd=root / 'source', env=environment(gpus, root),
                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            state.update(job_pid=process.pid, job_identity=process_identity(process.pid))
            state['training_started'] = state.get('training_started', False) or name != 'preflight'
            heartbeat()
            while process.poll() is None:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    heartbeat()
        state['jobs'][name] = {'completed': process.returncode == 0, 'exit_code': process.returncode}
        state.update(job_pid=None, job_identity=None)
        heartbeat()
        if process.returncode:
            raise RuntimeError(f'{name} failed with exit code {process.returncode}; inspect {log}')
    torchrun = [r.PYTHON, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2', '-m']
    heartbeat()
    try:
        execute('preflight', [*torchrun, MODULE + '.preflight', '--root', str(root)], root / 'preflight/console.log')
        if not json.loads((root / 'preflight.json').read_text())['passed']:
            raise RuntimeError('Two-GPU acceptance did not pass')
        state['gpu_acceptance_passed'] = True
        def train_registered(name):
            variant, seed_text = name.rsplit('_seed',1)
            directory = root/'explore'/name
            command=[*torchrun,MODULE+'.control','train','--root',str(root),
                     '--variant',variant,'--seed',seed_text,'--run-dir',str(directory)]
            if recover and (directory/'checkpoint.pth').exists():
                command.append('--recover')
            execute(name,command,directory/'console.log')
        for name in manifest['phase1']:
            train_registered(name)
        runs=r.collect_best_runs(root)
        promotion=r.choose_promotion(runs)
        report={'stage':'tcr_target_phase1','runs':runs,'promotion':promotion,
                'test_evaluated':False,'recommended_model':'c1','single_model_inference':True}
        r.write_json(root/'report.json',report)
        selected=promotion['selected_variant']
        seeds=(13,)
        replication=None
        if selected is not None:
            # Recipes and the gate are fixed before either seed13 run starts.
            for name in manifest['replication']:
                train_registered(name)
            seeds=r.SEEDS
            runs=r.collect_best_runs(root,seeds)
            replication=r.replication_gate(runs,selected)
            report.update(stage='tcr_target_replication',runs=runs,replication=replication)
            r.write_json(root/'report.json',report)
        names=[f'{v}_seed{seed}' for seed in seeds for v in r.VARIANTS]
        selections=freeze_test_selections(root,names,runs,selected)
        for name,item in selections.items():
            if any(r.sha256(item[key])!=item['sha256'] for key in ('checkpoint','original_checkpoint')):
                raise ValueError('Selected checkpoint changed')
            destination=root/'evaluation'/name
            execute('evaluate_'+name,[*torchrun,MODULE+'.evaluate','--checkpoint',item['checkpoint'],
                    '--output-root',str(destination),'--splits','val','test'],destination/'console.log')
        # Verify per-expression identity and independently recompute CPU IoU
        # before publishing any test summary. Validation must reproduce the
        # selected epoch; neither test result influences that selection.
        from script_eval.clip_differential_test import verify_result, aligned_samples
        evaluation={name:{split:json.loads((root/'evaluation'/name/f'{split}.json').read_text())
                          for split in ('val','test')} for name in names}
        verification={}
        for split,total in (('val',608),('test',2104)):
            aligned_samples({name:case[split]['samples'] for name,case in evaluation.items()},total)
            targets=None
            for name in names:
                current=[row['target_xywh'] for row in evaluation[name][split]['samples']]
                if targets is None:
                    targets=current
                elif current!=targets:
                    raise ValueError('Same-resolution evaluation received different ground truth coordinates')
                metadata={**selections[name],'validation':runs[name]}
                verification[f'{name}/{split}']=verify_result(evaluation[name][split],metadata,split,total)
                if any(r.sha256(selections[name][key])!=selections[name]['sha256']
                       for key in ('checkpoint','original_checkpoint')):
                    raise ValueError('Selected checkpoint changed during testing')
        r.write_json(root/'verification.json',{'passed':True,'results':verification,
            'original_and_frozen_checkpoints_unchanged':True,'test_used_for_selection':False})
        tests={name:evaluation[name]['test']['summary'] for name in names}
        test_groups={name:evaluation[name]['test']['groups'] for name in names}
        means={variant:sum(tests[f'{variant}_seed{seed}']['acc_05'] for seed in seeds)/len(seeds)
               for variant in r.VARIANTS}
        if selected is not None and replication is not None and replication['passed']:
            report['recommended_model']=selected
        report.update(stage='tcr_target_diagnostic_complete',test_evaluated=True,test=tests,
                      test_groups=test_groups,verification_passed=True,
                      test_seeds=list(seeds),mean_test_acc05_by_variant=means,
                      mean_test_target_reached_by_variant={k:v>=.74 for k,v in means.items()},
                      test_not_used_to_change_validation_selection=True)
        r.write_json(root/'report.json',report)
        state.update(phase='complete',current_job=None,test_evaluated=True,
                     mean_test_acc05_by_variant=means,
                     test_target_reached_by_variant={k:v>=.74 for k,v in means.items()})
    except BaseException as error:
        state.update(phase='failed', error=str(error))
        raise
    finally:
        state['controller_active'] = False
        heartbeat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'launch', 'resume', 'status', 'controller', 'train'))
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--gpus', default='auto')
    parser.add_argument('--cpu-report', type=Path)
    parser.add_argument('--variant', choices=r.VARIANTS, default='tcr_bias')
    parser.add_argument('--seed',type=int,choices=r.SEEDS,default=13)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--recover', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == 'prepare':
        prepare(root, args.cpu_report)
        print(root)
    elif args.action in ('launch', 'resume'):
        if not root.exists():
            prepare(root, args.cpu_report)
        if not launch(root, args.gpus, args.action == 'resume'):
            raise SystemExit(2)
    elif args.action == 'status':
        state = json.loads((root / 'status.json').read_text())
        state['live_processes'] = living_processes(state)
        print(json.dumps(state, indent=2, ensure_ascii=False))
    elif args.action == 'controller':
        controller(root, args.gpus, args.recover)
    else:
        r.train(root, args.run_dir.resolve(), args.variant, args.seed, args.recover)


if __name__ == '__main__':
    main()
