"""Detached resolution/reliability controls, validation gate and sealed replication."""
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

MODULE = 'script_train.clip_dense_grid_control'
DEFAULT_ROOT = r.REPOSITORY / 'output_clip_lavs_control/2026-10-31_dense_grid'
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
        'socket':'clip_dense_grid_'+root.name.replace('-','_'),
        'phase1':[f'{v}_seed13' for v in r.VARIANTS],
        'replication':[f'{v}_seed{seed}' for seed in (14,15) for v in r.VARIANTS],
        'epochs_per_run':r.EPOCHS,'maximum_training_runs':9,'maximum_epochs':9*r.EPOCHS,
        'selection':'first strict best validation Acc05 including resolution-specific initial; raw wins same-epoch EMA ties',
        'initialization':'paired original C1 seeds, not test-selected seed or latest adaptation',
        'method':{'resolutions':[224,336,336],'visual_grids':[14,21,21],
                  'new_reliability_parameters':257,'inference_models':1,'extra_inference_views':0,
                  'effective_batch':96,'microbatch_per_gpu':4,'accumulation':12,
                  'vision_lora_lr':4e-6,'fusion_decoder_lr':2e-6,'reliability_lr':1e-4,
                  'native_pixels_resized_directly':True,'test_driven_selection':False,
                  'novelty':'higher resolution and parameter averaging are controls; foreground-relative text reliability needs ablation and external comparison'},
        'test_used_for_selection':False,'test_previously_observed':True,
        'test_only_after_validation_screen_and_replication_pass':True,
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
    if subprocess.run(['tmux', '-L', manifest['socket'], 'has-session', '-t', 'suite'],
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
        raise RuntimeError('An existing tmux session owns this experiment')
    command = [r.PYTHON, '-u', '-m', MODULE + '.control', 'controller', '--root', str(root), '--gpus', gpus]
    if resume:
        command.append('--recover')
    shell = (f'cd {shlex.quote(str(root / "source"))} && {shlex.join(command)} '
             f'>> {shlex.quote(str(root / "controller.log"))} 2>&1 < /dev/null; '
             'reobserve_rc=$?; exit "$reobserve_rc"')
    subprocess.run(['tmux', '-L', manifest['socket'], '-f', '/dev/null', 'new-session', '-d', '-s', 'suite', shell],
                   check=True, env=environment(gpus, root), stdin=subprocess.DEVNULL,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, start_new_session=True)
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
        if job in manifest['phase1']:
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
        report={'stage':'dense_grid_phase1','runs':runs,'promotion':promotion,
                'test_evaluated':False,'recommended_model':'c1','single_model_inference':True}
        r.write_json(root/'report.json',report)
        selected=promotion['selected_variant']
        if selected is None:
            state.update(phase='gate_not_passed',current_job=None,test_evaluated=False)
            return
        # All phase-two recipes were registered before any phase-one training.
        for name in manifest['replication']:
            train_registered(name)
        runs=r.collect_best_runs(root,r.SEEDS)
        replication=r.replication_gate(runs,selected)
        report.update(stage='dense_grid_replication',runs=runs,replication=replication)
        r.write_json(root/'report.json',report)
        if not replication['passed']:
            state.update(phase='complete_no_stable_gain',current_job=None,test_evaluated=False)
            return
        views=('c1_224','dense_336') if selected=='dense_336' else r.VARIANTS
        names=[f'{v}_seed{seed}' for seed in r.SEEDS for v in views]
        selections={name:{'checkpoint':str(root/'explore'/name/'best_checkpoint.pth'),
                          'sha256':r.sha256(root/'explore'/name/'best_checkpoint.pth'),
                          'epoch':runs[name]['epoch'],'weight_view':runs[name]['weight_view']}
                    for name in names}
        registration=root/'test_registration.json'
        if registration.exists():
            if json.loads(registration.read_text())['selected']!=selections:
                raise ValueError('Sealed test selection changed')
        else:
            r.write_json(registration,{'selected':selections,'selection_used_test':False,
                                       'selected_variant':selected,'created_at':time.time()})
        for name,item in selections.items():
            if r.sha256(item['checkpoint'])!=item['sha256']:
                raise ValueError('Selected checkpoint changed')
            destination=root/'evaluation'/name
            execute('evaluate_'+name,[*torchrun,MODULE+'.evaluate','--checkpoint',item['checkpoint'],
                    '--output-root',str(destination),'--splits','val','test'],destination/'console.log')
        tests={name:json.loads((root/'evaluation'/name/'test.json').read_text())['summary'] for name in names}
        mean_acc=sum(tests[f'{selected}_seed{seed}']['acc_05'] for seed in r.SEEDS)/len(r.SEEDS)
        report.update(test_evaluated=True,test=tests,mean_test_acc05=mean_acc,
                      mean_test_target_reached=mean_acc>=.74,recommended_model=selected,
                      test_not_used_to_change_validation_selection=True)
        r.write_json(root/'report.json',report)
        state.update(phase='complete',current_job=None,test_evaluated=True,
                     mean_test_acc05=mean_acc,test_target_reached=mean_acc>=.74)
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
    parser.add_argument('--variant', choices=r.VARIANTS, default='dense_text_336')
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
