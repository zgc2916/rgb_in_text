"""Immutable, detached spatial center and utility screening; validation only."""
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

MODULE = 'script_train.clip_spatial_quality_control'
DEFAULT_ROOT = r.REPOSITORY / 'output_clip_lavs_control/2026-10-27_spatial_quality'
TERMINAL = ('gate_not_passed', 'phase1_complete')


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def prepare(root, cpu_report=None):
    if root.exists():
        raise FileExistsError('Experiment exists; use launch or resume with its frozen source')
    config = r.recipe(root)
    resources = [Path(config.exchange_base_checkpoint)]
    resources += [p for p in Path(config.pretrained_model_path).rglob('*') if p.is_file()]
    resources += [p for p in Path(config.dataset_bert_model_path).rglob('*') if p.is_file() and p.stat().st_size < 10 * 1024**2]
    resources += [Path(config.split_root) / config.dataset / f'{config.dataset}_{split}.pth' for split in ('train', 'val')]
    resources += [r.REFERENCE / 'evaluation/c1_seed13/val.json']
    for path in resources:
        if not path.is_file():
            raise FileNotFoundError(path)
    if not Path(config.pretrained_model_path, 'pytorch_model.bin').is_file():
        raise FileNotFoundError('Local CLIP weights are required; downloads are disabled')
    root.mkdir(parents=True)
    source = root / 'source'
    source.mkdir()
    for name in ('models', 'datasets', 'utils', 'train_val', 'script_train', 'script_eval', 'tests'):
        shutil.copytree(r.REPOSITORY / name, source / name,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.pytest_cache'))
    shutil.copy2(r.REPOSITORY / 'engine.py', source / 'engine.py')
    r.write_json(source / 'run_resources/c1_recipe.json',
                 json.loads((r.REFERENCE / 'formal/c1_seed13/recipe.json').read_text()))
    r.write_json(source / 'run_resources/c1_references.json', r.references())
    files = {str(p.relative_to(source)): r.sha256(p) for p in sorted(source.rglob('*')) if p.is_file()}
    r.write_json(root / 'source_manifest.json', files)
    r.write_json(root / 'resource_manifest.json', {str(p.resolve()): r.sha256(p) for p in resources})
    recipes = {f'{v}_seed13': vars(r.recipe(root / 'explore' / f'{v}_seed13', v)) for v in r.VARIANTS}
    provenance = {'source_sha256': r.sha256(root / 'source_manifest.json'),
                  'resources_sha256': r.sha256(root / 'resource_manifest.json'),
                  'recipes_sha256': object_hash(recipes), 'python': r.PYTHON}
    manifest = {'version': 1, 'provenance': provenance, 'recipes': recipes,
        'references': [r.reference(13)], 'socket': 'clip_spatial_' + root.name.replace('-', '_'),
        'phase1': list(recipes), 'epochs_per_run': r.EPOCHS, 'maximum_training_runs': 4,
        'selection': 'first strict maximum validation Acc@0.5 including C1 at epoch -1',
        'diagnostic_selection': 'also save the first strict best trained checkpoint separately',
        'phase1_gate': {'extra_hits_vs_original': 4, 'extra_hits_vs_continuation': 2,
                        'quality_non_decrease': ['accu_07', 'miou', 'miou_weak', 'accu_normal'],
                        'calibrated_utility_must_beat_uniform_spatial_reading': True},
        'method': {'kind':'text-conditioned local center distributions with relative-IoU utility',
            'views_per_sample':2,'local_encoder_input':224,'shared_CLIP_weights':True,
            'width_height_policy':'retain coarse box dimensions in all local variants',
            'spatial_supervision':'GT center Gaussian on the local patch grid; training loss only',
            'utility_supervision':'stop-gradient per-modality IoU gain relative to current coarse box',
            'fusion':'B uniform global/RGB/TIR; C/D accept only predicted positive local benefit',
            'adaptive_reading':'D only; B/C use the same fixed native crop',
            'hidden_dimension':128,'coarse_crop_scale':3.,'coarse_crop_floor_224_pixels':64,
            'native_pixels_preserved_before_global_resize':True,'crop_policy_uses_GT':False,
            'zero_spatial_score_initialization':True,
            'base_trainable':['last VL encoder layer','final VL norm','bbox head'],
            'frozen':['both CLIP/LoRA branches','external LAVS','IAFv3','other VL layers'],
            'effective_batch':96,'microbatch_per_gpu':4,'gradient_accumulation':12,
            'existing_lr':2e-6,'new_lr':1e-4,
            'gradient_clipping':'separate base, spatial reader and utility-head groups at norm1',
            'gradient_clipping_and_base_lr_shared_by_all_controls':True,
            'scope':'trained-C1 warm-start screen; all runs rebuild original C1 independently',
            'novelty':'unverified RGB/TIR grounding adaptation; spatial moments and quality estimation have prior art',
            'sources':['https://arxiv.org/abs/1801.07372',
                       'https://openaccess.thecvf.com/content/CVPR2021/papers/Li_Generalized_Focal_Loss_V2_Learning_Reliable_Localization_Quality_Estimation_for_CVPR_2021_paper.pdf']},
        'test_used_for_selection': False, 'automatic_test_evaluation': False,
        'test_previously_observed': True, 'automatic_additional_seeds': False,
        'gpu_acceptance_required': True, 'checkpoint_recovery': ['optimizer', 'scheduler', 'all_rank_RNG', 'selection_journals']}
    if cpu_report is not None:
        data = json.loads(cpu_report.read_text())
        if not data['passed'] or set(data['variants']) != set(r.VARIANTS):
            raise ValueError('CPU acceptance did not pass all variants')
        if any(r.sha256(r.REPOSITORY / name) != digest for name, digest in data['source_sha256'].items()):
            raise ValueError('Source changed after CPU acceptance')
        shutil.copy2(cpu_report, root / 'cpu_acceptance.json')
        manifest['cpu_acceptance_sha256'] = r.sha256(root / 'cpu_acceptance.json')
    r.write_json(root / 'preregistration.json', manifest)
    r.write_json(root / 'status.json', {'phase': 'prepared', 'training_started': False,
        'completed_epochs': 0, 'controller_active': False, 'source_snapshot': str(source),
        'cpu_acceptance_passed': cpu_report is not None, 'gpu_acceptance_passed': False, 'jobs': {}})
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
        state.update(phase='gpu_preflight' if name == 'preflight' else 'training_controls',
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
        for variant in r.VARIANTS:
            name = variant + '_seed13'
            directory = root / 'explore' / name
            command = [*torchrun, MODULE + '.control', 'train', '--root', str(root),
                       '--variant', variant, '--run-dir', str(directory)]
            if recover and (directory / 'checkpoint.pth').exists():
                command.append('--recover')
            execute(name, command, directory / 'console.log')
        from script_train.clip_reliability_control.runtime import best_from_log
        runs = {v: best_from_log(root / 'explore' / f'{v}_seed13') for v in r.VARIANTS}
        promotion = r.choose_promotion(manifest['references'][0]['validation'], runs)
        report = {'stage': 'single_seed_warm_start_screening', 'runs': runs,
                  'best_trained': {v: json.loads((root / 'explore' / f'{v}_seed13/trained_best.json').read_text())
                                   for v in r.VARIANTS}, 'promotion': promotion,
                  'test_evaluated': False, 'multi_seed_replication_completed': False,
                  'recommended_model': 'c1_pending_replication' if promotion['selected_variant'] else 'c1'}
        r.write_json(root / 'report.json', report)
        state.update(phase='phase1_complete' if promotion['selected_variant'] else 'gate_not_passed',
                     current_job=None, completed_epochs=len(r.VARIANTS) * r.EPOCHS,
                     recommended_model=report['recommended_model'])
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
    parser.add_argument('--variant', choices=r.VARIANTS, default='anchor_text_separate')
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
        r.train(root, args.run_dir.resolve(), args.variant, 13, args.recover)


if __name__ == '__main__':
    main()
