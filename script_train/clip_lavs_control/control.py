"""Persistent tmux suite. Usage: python -m ...control launch|status|attach|resume."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import threading
import time
from .runtime import (REPOSITORY, REFERENCE, PYTHON, VARIANTS, recipe, sha256, write_json)

DEFAULT_ROOT = REPOSITORY/'output_clip_lavs_control/2026-10-09_text_local_exchange'


def process_identity(pid):
    try:
        raw = Path(f'/proc/{pid}/stat').read_text()
        return raw[raw.rfind(')')+2:].split()[19]
    except (FileNotFoundError, ProcessLookupError):
        return None


def living_processes(state):
    found = []
    for label, pid_key, identity_key in (('controller', 'controller_pid', 'controller_identity'),
                                         ('training', 'job_pid', 'job_identity')):
        pid = state.get(pid_key)
        if pid and process_identity(pid) == state.get(identity_key):
            found.append((label, pid))
    return found


def environment(gpus, root):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=gpus, OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
               OPENBLAS_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1',
               MPLCONFIGDIR=str(root/'matplotlib'))
    return env


def snapshot(root, gpus):
    if root.exists():
        raise FileExistsError('Experiment exists; use resume for its immutable snapshot')
    root.mkdir(parents=True)
    destination = root/'source'
    destination.mkdir()
    for name in ('models', 'datasets', 'utils', 'train_val', 'script_train', 'script_eval'):
        shutil.copytree(REPOSITORY/name, destination/name,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.pytest_cache'))
    shutil.copy2(REPOSITORY/'engine.py', destination/'engine.py')
    write_json(destination/'run_resources/c1_recipe.json',
               json.loads((REFERENCE/'preregistration.json').read_text())['recipes']['formal/c1_seed13'])
    files = {str(p.relative_to(destination)): sha256(p) for p in sorted(destination.rglob('*')) if p.is_file()}
    write_json(root/'source_manifest.json', files)
    source_hash = sha256(root/'source_manifest.json')
    init = recipe(root).retrain
    initial_hash = sha256(init)
    config = recipe(root)
    # Preserve small initialization metadata and hash all external runtime
    # resources that determine the scientific setup; never copy image data.
    resources = {}
    for path in sorted(Path(config.pretrained_model_path).rglob('*')):
        if path.is_file():
            resources[str(path.resolve())] = sha256(path)
            if path.stat().st_size < 10*1024**2:
                target = root/'resources/clip'/path.relative_to(config.pretrained_model_path)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
    for split in ('train', 'val', 'test'):
        path = Path(config.split_root)/config.dataset/f'{config.dataset}_{split}.pth'
        resources[str(path.resolve())] = sha256(path)
        target = root/'resources/annotations'/path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    for path in sorted(Path(config.dataset_bert_model_path).rglob('*')):
        if path.is_file() and path.stat().st_size < 10*1024**2:
            resources[str(path.resolve())] = sha256(path)
            target = root/'resources/bert'/path.relative_to(config.dataset_bert_model_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    write_json(root/'resource_manifest.json', resources)
    selected = [x for x in json.loads((REFERENCE/'selected_models.json').read_text()) if x['variant'] == 'c1']
    recipes = {f'{v}_seed{s}': vars(recipe(root/'formal'/f'{v}_seed{s}', v, s)) for v in VARIANTS for s in (13, 14, 15)}
    provenance = {'source_manifest_sha256': source_hash, 'initialization_sha256': initial_hash,
                  'resource_manifest_sha256': sha256(root/'resource_manifest.json'),
                  'gpus': gpus, 'python': PYTHON,
                  'recipes_sha256': __import__('hashlib').sha256(json.dumps(recipes, sort_keys=True).encode()).hexdigest()}
    socket = 'clip_lavs_'+root.name.replace('-', '_')
    write_json(root/'preregistration.json', {'version': 2, 'provenance': provenance,
        'initialization_sha256': initial_hash, 'source_manifest_sha256': source_hash,
        'recipes': recipes, 'references': selected, 'inherited_missing_keys': [],
        'gpus': gpus, 'python': PYTHON, 'socket': socket, 'session': 'suite',
        'phase1': [f'{v}_seed13' for v in VARIANTS], 'maximum_new_full_runs': 6,
        'selection': 'first strictly maximal validation Acc@0.5; initial audits excluded',
        'gate': {'c1_acc05_minimum_delta': .005, 'minimum_additional_hits': 4,
                 'c1_acc07_miou_non_decrease': True, 'text_mechanism_requires_local_dominance': True},
        'test_selection': False, 'test_previously_observed': True,
        'method': {'replaced': 'external RGB/TIR exchange only',
            'preserved': ['internal language bridges', 'IAFv3', 'six VL encoder layers'],
            'radii': [1, 1, 2, 3], 'text_off_ablation_parameter_matched': True,
            'novelty_status': 'hypothesis requiring controlled evidence',
            'sources': ['https://arxiv.org/abs/2604.18829', 'https://arxiv.org/abs/2605.21882',
                'https://proceedings.iclr.cc/paper_files/paper/2025/file/b9079234614422c036c5a92bb8ec04c4-Paper-Conference.pdf']}, 'bootstrap_unit': 'image', 'bootstrap_repeats': 5000})


def verify_snapshot(root):
    manifest = json.loads((root/'preregistration.json').read_text())
    files = json.loads((root/'source_manifest.json').read_text())
    if sha256(root/'source_manifest.json') != manifest['source_manifest_sha256']:
        raise ValueError('Source manifest changed')
    for name, expected in files.items():
        if sha256(root/'source'/name) != expected:
            raise ValueError(f'Frozen source changed: {name}')
    recipes_hash = __import__('hashlib').sha256(json.dumps(manifest['recipes'], sort_keys=True).encode()).hexdigest()
    if recipes_hash != manifest['provenance']['recipes_sha256']:
        raise ValueError('Frozen recipe configuration changed')
    if manifest['gpus'] != manifest['provenance']['gpus'] or manifest['python'] != manifest['provenance']['python']:
        raise ValueError('Frozen runtime configuration changed')
    if sha256(recipe(root).retrain) != manifest['initialization_sha256']:
        raise ValueError('Public initialization changed')
    if sha256(root/'resource_manifest.json') != manifest['provenance']['resource_manifest_sha256']:
        raise ValueError('Resource manifest changed')
    for path, expected in json.loads((root/'resource_manifest.json').read_text()).items():
        if sha256(path) != expected:
            raise ValueError(f'External runtime resource changed: {path}')
    for item in manifest['references']:
        if sha256(item['checkpoint']) != item['sha256']:
            raise ValueError('C1 reference changed')
    return manifest


def launch(root, gpus, resume=False):
    if resume:
        manifest = verify_snapshot(root)
        if gpus != manifest['gpus']:
            raise ValueError('Recovery GPU configuration differs')
        state = json.loads((root/'status.json').read_text()) if (root/'status.json').exists() else {}
        if living_processes(state):
            raise RuntimeError(f'Refusing duplicate controller/training: {living_processes(state)}')
        if state.get('phase') in ('complete', 'gate_not_passed'):
            raise RuntimeError('Suite already finished; no recovery needed')
    else:
        snapshot(root, gpus)
        manifest = verify_snapshot(root)
    # Do not recreate a session that already owns processes even if status is stale.
    exists = subprocess.run(['tmux', '-L', manifest['socket'], 'has-session', '-t', 'suite'],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    if exists:
        raise RuntimeError('tmux session already exists; refusing duplicate training')
    command = [PYTHON, '-u', '-m', 'script_train.clip_lavs_control.control', 'controller', '--root', str(root), '--gpus', gpus]
    if resume:
        command.append('--recover')
    # Keep a shell attached to tmux's PTY while the controller's stdio is
    # redirected. exec would close every slave FD and tmux could retire the
    # pane before the controller writes its first status.
    shell = (f'cd {shlex.quote(str(root / "source"))} && {shlex.join(command)} '
             f'>> {shlex.quote(str(root / "controller.log"))} 2>&1 < /dev/null; '
             'clip_lavs_rc=$?; exit "$clip_lavs_rc"')
    subprocess.run(['tmux', '-L', manifest['socket'], '-f', '/dev/null', 'new-session', '-d', '-s', 'suite', shell],
                   check=True, env=environment(gpus, root), stdin=subprocess.DEVNULL,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, start_new_session=True)
    for _ in range(30):
        time.sleep(1)
        if (root/'status.json').exists():
            state = json.loads((root/'status.json').read_text())
            if state.get('phase') == 'failed':
                raise RuntimeError(state.get('error'))
            if any(label == 'controller' for label, _ in living_processes(state)):
                print(json.dumps({'launched': True, 'root': str(root), 'status': state}, indent=2), flush=True)
                return
    raise RuntimeError('Controller did not become healthy within 30 seconds; inspect controller.log')


def gpu_check(gpus):
    result = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,memory.free',
                                      '--format=csv,noheader,nounits'], text=True)
    available = {int(row.split(',')[0]): tuple(int(x) for x in row.split(',')[1:]) for row in result.splitlines()}
    for index in map(int, gpus.split(',')):
        used, free = available[index]
        if used > 2048 or free < 18000:
            raise RuntimeError(f'GPU {index} unavailable: used={used}MB free={free}MB')


def first_best(directory):
    rows = [json.loads(x) for x in (directory/'log.txt').read_text().splitlines() if x.strip()]
    if [r['epoch'] for r in rows] != list(range(120)):
        raise ValueError(f'Incomplete full run: {directory}')
    index = max(range(len(rows)), key=lambda i: rows[i]['val_accu'])
    stored = json.loads((directory/'best.json').read_text())
    if stored['epoch'] != rows[index]['epoch']:
        raise ValueError('Best checkpoint selection differs from first strict maximum')
    return stored


def choose_promotion(baseline, runs):
    checks, eligible = {}, []
    for variant in VARIANTS:
        candidate = runs[variant]
        additional = round(candidate['accu'] * 608) - round(baseline['accu'] * 608)
        passed = (additional >= 4 and candidate['accu'] - baseline['accu'] >= .005 - 1e-12
                  and all(candidate[k] >= baseline[k] - 1e-12 for k in ('accu_07', 'miou')))
        checks[variant] = {'additional_hits': additional, 'c1_gate': passed}
        if passed:
            eligible.append(variant)
    selected = max(eligible, key=lambda v: (runs[v]['accu'], runs[v]['accu_07'], runs[v]['miou'])) if eligible else None
    text, local = runs['text_local'], runs['local']
    mechanism = (all(text[k] >= local[k] - 1e-12 for k in ('accu', 'accu_07', 'miou'))
                 and any(text[k] > local[k] + 1e-12 for k in ('accu', 'accu_07', 'miou')))
    return {'selected_variant': selected, 'checks': checks,
            'text_gate_dominates_local': mechanism, 'test_used': False}


def controller(root, gpus, recover=False):
    lock = (root/'suite.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = json.loads((root/'preregistration.json').read_text())
    state = json.loads((root/'status.json').read_text()) if recover and (root/'status.json').exists() else {'jobs': {}}
    if state.get('job_pid') and process_identity(state['job_pid']) == state.get('job_identity'):
        raise RuntimeError('Previous torchrun is still alive; refusing duplicate recovery')
    state.update(controller_pid=os.getpid(), controller_identity=process_identity(os.getpid()),
                 socket=manifest['socket'], session=manifest['session'], phase='preflight',
                 source_snapshot=str(root/'source'), started_at=time.time(), error=None)
    heartbeat_lock = threading.Lock()
    def heartbeat():
        with heartbeat_lock:
            current = state.get('current_job', '')
            if current and current.startswith('formal/'):
                progress = root/'formal'/current.split('/', 1)[1]/'progress.json'
                if progress.exists():
                    state.update(json.loads(progress.read_text()))
            state['heartbeat_at'] = time.time()
            write_json(root/'status.json', state)
    stopped = threading.Event()
    def pulse():
        while not stopped.wait(30):
            heartbeat()
    heartbeat()
    threading.Thread(target=pulse, daemon=True).start()
    torchrun = [PYTHON, '-u', '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2', '-m']
    def execute(name, command, log):
        if state['jobs'].get(name, {}).get('completed'):
            return
        previous_phase = state['phase']
        while True:
            try:
                gpu_check(gpus)
                break
            except RuntimeError as error:
                state.update(phase='waiting_for_gpus', current_job=name, waiting_reason=str(error))
                heartbeat()
                time.sleep(30)
        state.update(phase=previous_phase, current_job=name, epoch=None, best=None, exit_code=None, waiting_reason=None)
        log.parent.mkdir(parents=True, exist_ok=True)
        write_json(log.with_suffix('.command.json'), {'argv': command, 'gpus': gpus, 'cwd': str(root/'source')})
        with log.open('a') as stream:
            process = subprocess.Popen(command, cwd=root/'source', env=environment(gpus, root),
                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            state.update(job_pid=process.pid, job_identity=process_identity(process.pid))
            heartbeat()
            while process.poll() is None:
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    heartbeat()
            state['exit_code'] = process.returncode
            state['jobs'][name] = {'completed': process.returncode == 0, 'exit_code': process.returncode, 'finished_at': time.time()}
            heartbeat()
            if process.returncode:
                raise RuntimeError(f'{name} failed with exit code {process.returncode}; inspect {log}')
    try:
        manifest = verify_snapshot(root)
        for item in manifest['references']:
            name = f"c1_seed{item['seed']}"
            destination = root/'references'/name
            if not (destination/'val.json').exists():
                execute('reference/'+name, [*torchrun, 'script_train.clip_lavs_control.evaluate',
                    '--checkpoint', item['checkpoint'], '--output-root', str(destination), '--splits', 'val'],
                    destination/'console.log')
            compare_reference(root, item)
        execute('preflight', [*torchrun, 'script_train.clip_lavs_control.control', 'preflight', '--root', str(root)],
                root/'preflight/console.log')
        audit = json.loads((root/'preflight.json').read_text())
        manifest['inherited_missing_keys'] = audit['inherited_missing_keys']
        manifest['inherited_extra_keys'] = audit['inherited_extra_keys']
        write_json(root/'preregistration.json', manifest)
        state['phase'] = 'phase1'
        heartbeat()
        for variant in VARIANTS:
            formal_job(root, variant, 13, recover, execute, torchrun)
        runs = {v: first_best(root/'formal'/f'{v}_seed13') for v in VARIANTS}
        baseline = next(x['validation'] for x in manifest['references'] if x['seed'] == 13)
        gate = choose_promotion(baseline, runs)
        write_json(root/'promotion.json', gate)
        names = [(v, 13) for v in VARIANTS]
        if gate['selected_variant']:
            state['phase'] = 'phase2'
            heartbeat()
            # Repeat BOTH variants, preserving the matched mechanism ablation.
            for seed in (14, 15):
                for variant in VARIANTS:
                    formal_job(root, variant, seed, recover, execute, torchrun)
                    names.append((variant, seed))
        selected = freeze(root, names)
        state['phase'] = 'frozen_validation'
        heartbeat()
        for item in selected:
            out = root/'evaluation'/item['name']
            if not (out/'val.json').exists():
                execute('validation/'+item['name'], [*torchrun, 'script_train.clip_lavs_control.evaluate',
                    '--checkpoint', item['checkpoint'], '--output-root', str(out), '--splits', 'val'], out/'val.console.log')
        if gate['selected_variant']:
            state['phase'] = 'frozen_test'
            write_json(root/'test_registration.json', {'models': selected, 'frozen_before_test': True,
                       'test_previously_observed': True, 'timestamp': time.time()})
            heartbeat()
            for item in selected:
                if sha256(item['checkpoint']) != item['sha256']:
                    raise ValueError('Frozen checkpoint changed')
                out = root/'evaluation'/item['name']
                if not (out/'test.json').exists():
                    execute('test/'+item['name'], [*torchrun, 'script_train.clip_lavs_control.evaluate',
                        '--checkpoint', item['checkpoint'], '--output-root', str(out), '--splits', 'test'], out/'test.console.log')
        from .report import report
        write_json(root/'report.json', report(root))
        state.update(phase='complete' if gate['selected_variant'] else 'gate_not_passed',
                     current_job=None, finished_at=time.time())
        heartbeat()
    except BaseException as error:
        state.update(phase='failed', error=str(error), failed_at=time.time())
        heartbeat()
        raise
    finally:
        stopped.set()

def formal_job(root, variant, seed, recover, execute, torchrun):
    name = f'{variant}_seed{seed}'
    directory = root/'formal'/name
    command = [*torchrun, 'script_train.clip_lavs_control.control', 'train', '--root', str(root),
               '--run-dir', str(directory), '--variant', variant, '--seed', str(seed)]
    if recover and (directory/'checkpoint.pth').exists():
        command.append('--recover')
    execute('formal/'+name, command, directory/'console.log')


def compare_reference(root, item):
    import torch
    import numpy as np
    name = f"c1_seed{item['seed']}"
    old = json.loads((REFERENCE/'evaluation'/name/'val.json').read_text())
    new = json.loads((root/'references'/name/'val.json').read_text())
    if len(old['samples']) != len(new['samples']):
        raise AssertionError('Reference evaluation incomplete')
    for a, b in zip(old['samples'], new['samples']):
        if any(a[k] != b[k] for k in ('index', 'image', 'phrase', 'target_xywh')):
            raise AssertionError('Reference annotation identity differs')
    delta = float(np.abs(np.array([x['pred_xywh'] for x in old['samples']])-np.array([x['pred_xywh'] for x in new['samples']])).max())
    if delta > 1e-6 or any(new['summary'][k] != old['summary'][k] for k in ('correct_05', 'correct_07')):
        raise AssertionError(f'Legacy C1 reproduction differs: {name}, max error={delta}')
    checkpoint = torch.load(item['checkpoint'], map_location='cpu', weights_only=False)
    stored = vars(checkpoint['args'])
    expected = vars(recipe(root, seed=item['seed'], backend='legacy'))
    fields = ('model', 'FusionMethod', 'dataset', 'imsize', 'batch_size', 'grad_accum_steps',
              'lr', 'weight_decay', 'target_safe_crop', 'target_scale_floor_pixels', 'lora_r_rgb',
              'lora_r_ir', 'match_infmae_lora_schedule', 'seed', 'retrain', 'image_norm')
    if any(stored[k] != expected[k] for k in fields):
        raise AssertionError('Reference scientific recipe differs')
    write_json(root/'references'/name/'audit.json', {'passed': True, 'max_forward_error': delta,
               'checkpoint_sha256': item['sha256'], 'recipe_fields_checked': fields, 'summary': new['summary']})


def freeze(root, names):
    import torch
    selected = []
    for variant, seed in names:
        name = f'{variant}_seed{seed}'
        directory = root/'formal'/name
        best = first_best(directory)
        out = root/'frozen'/f'{name}.pth'
        if not out.exists():
            checkpoint = torch.load(directory/'best_checkpoint.pth', map_location='cpu', weights_only=False)
            if checkpoint['epoch'] != best['epoch']:
                raise AssertionError('Best checkpoint epoch mismatch')
            from .runtime import atomic_checkpoint
            atomic_checkpoint(out, checkpoint)
        selected.append({'name': name, 'variant': variant, 'seed': seed, 'epoch': best['epoch'],
                         'checkpoint': str(out), 'sha256': sha256(out), 'validation': best})
    previous = root/'selected_models.json'
    if previous.exists() and json.loads(previous.read_text()) != selected:
        raise ValueError('Frozen registration changed')
    write_json(previous, selected)
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('launch', 'status', 'attach', 'resume', 'controller', 'train', 'preflight'))
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--gpus')
    parser.add_argument('--recover', action='store_true')
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--variant', choices=VARIANTS, default='text_local')
    parser.add_argument('--seed', type=int, default=13)
    cli = parser.parse_args()
    root = cli.root.resolve()
    if cli.gpus is None:
        cli.gpus = (json.loads((root/'preregistration.json').read_text())['gpus']
                    if cli.action == 'resume' else '6,7')
    if cli.action in ('launch', 'resume'):
        launch(root, cli.gpus, cli.action == 'resume')
    elif cli.action == 'status':
        state = json.loads((root/'status.json').read_text())
        state['live_processes'] = living_processes(state)
        print(json.dumps(state, indent=2, ensure_ascii=False))
    elif cli.action == 'attach':
        manifest = json.loads((root/'preregistration.json').read_text())
        os.execvp('tmux', ['tmux', '-L', manifest['socket'], 'attach-session', '-t', manifest['session']])
    elif cli.action == 'controller':
        controller(root, cli.gpus, cli.recover)
    elif cli.action == 'train':
        from .runtime import train
        train(root, cli.run_dir.resolve(), cli.variant, cli.seed, cli.recover)
    else:
        from .preflight import run
        run(root)


if __name__ == '__main__':
    main()
