"""User-authorized diagnostic tests after the differential training queue.

Run a frozen copy of this file with PYTHONPATH pointing at training/source.
The evaluator is separately copied and hashed: its corrected absolute imports
load all models, data and checkpoint restoration from that unchanged snapshot.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from script_eval.clip_dense_grid_test import (
    sha256, write_json, phase_one_ready, aligned_samples, summarize, paired_changes,
)

TERMINAL = ('complete', 'gate_not_passed', 'complete_no_stable_gain')


def training_ready(root, names, epochs, state):
    if not phase_one_ready(root, names, epochs) or state.get('controller_active', True):
        return False
    if state.get('phase') in TERMINAL:
        return True
    # A frozen historical evaluator can fail after all training completed.
    # This independent, validation-preflighted evaluator handles that case.
    return (state.get('phase') == 'failed'
            and str(state.get('current_job', '')).startswith('evaluate_'))


def verify_request(root, output):
    request = json.loads((output/'request.json').read_text())
    if Path(request['training_root']).resolve() != root:
        raise ValueError('Request belongs to another training suite')
    for item in request['code'].values():
        if sha256(item['path']) != item['sha256']:
            raise ValueError('Registered diagnostic code changed')
    if sha256(Path(__file__)) != request['code']['runner']['sha256']:
        raise ValueError('Runner differs from the frozen request')
    return request


def completed_seeds(root, runtime):
    seeds = []
    for seed in runtime.SEEDS:
        names = [f'{v}_seed{seed}' for v in runtime.VARIANTS]
        if phase_one_ready(root, names, runtime.EPOCHS):
            seeds.append(seed)
        elif any((root/'explore'/name/'progress.json').exists() for name in names):
            raise ValueError(f'Terminal training has incomplete seed {seed}')
    if 13 not in seeds:
        raise ValueError('Seed13 controls are incomplete')
    return seeds


def seal_plan(root, output, registration, runtime, state, request):
    seeds = completed_seeds(root, runtime)
    selected = runtime.collect_best_runs(root, seeds)
    plan_path = output/'evaluation_plan.json'
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        if set(plan['checkpoints']) != set(selected):
            raise ValueError('Sealed training selections changed')
        for name, item in plan['checkpoints'].items():
            # Validation metadata contains a historical NaN mask metric.
            # Check its stable serialized representation rather than NaN==NaN.
            if json.dumps(item['validation'], sort_keys=True) != json.dumps(selected[name], sort_keys=True):
                raise ValueError('Sealed validation metadata changed')
        return plan
    cases = {}
    for name, validation in selected.items():
        original = root/'explore'/name/'best_checkpoint.pth'
        digest = sha256(original)
        frozen = output/'frozen'/f'{name}.pth'
        frozen.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, frozen)
        if sha256(original) != digest or sha256(frozen) != digest:
            raise ValueError('Checkpoint changed while sealing test selections')
        cases[name] = {'checkpoint': str(frozen), 'original_checkpoint': str(original),
                       'sha256': digest, 'epoch': validation['epoch'],
                       'weight_view': validation['weight_view'], 'validation': validation}
    args = runtime.recipe(output, 'lavs_control', 13)
    annotations = {split: Path(args.split_root)/args.dataset/f'{args.dataset}_{split}.pth'
                   for split in ('val', 'test')}
    plan = {'training_root': str(root), 'seeds': seeds, 'checkpoints': cases,
            'created_at': time.time(), 'request_sha256': sha256(output/'request.json'),
            'source_manifest_sha256': registration['provenance']['source_sha256'],
            'training_terminal_phase': state['phase'],
            'training_error': state.get('error'),
            'annotations': {key: {'path': str(path), 'sha256': sha256(path)} for key,path in annotations.items()},
            'selection': f'First strict best validation Acc05 after all {runtime.EPOCHS} epochs, including initial; raw wins same-epoch ties',
            'user_requested_diagnostic': True, 'test_used_for_selection': False,
            'original_training_gate_unchanged': True,
            'protocol': {'world_size': 2, 'batch_size_per_gpu': 12, 'single_model_inference': True,
                         'splits': ['val', 'test'], 'ground_truth_in_forward': False,
                         'grad_context': 'torch.no_grad', 'optimizer_created': False},
            'code': request['code']}
    write_json(plan_path, plan)
    return plan


def verify_plan(root, output, plan):
    if sha256(output/'request.json') != plan['request_sha256']:
        raise ValueError('Evaluation request changed after selections were sealed')
    if sha256(root/'source_manifest.json') != plan['source_manifest_sha256']:
        raise ValueError('Training source manifest changed')
    for item in plan['annotations'].values():
        if sha256(item['path']) != item['sha256']:
            raise ValueError('Annotations changed')
    for item in plan['checkpoints'].values():
        if any(sha256(item[key]) != item['sha256'] for key in ('checkpoint', 'original_checkpoint')):
            raise ValueError('Selected or frozen checkpoint changed')


def verify_result(data, selected, split, total):
    import torch
    from utils.box_utils import bbox_iou, xywh2xyxy
    source = data['source']
    if (source['sha256'] != selected['sha256'] or source['epoch'] != selected['epoch']
            or source['weight_view'] != selected['weight_view'] or data['split'] != split):
        raise ValueError('Evaluation does not belong to its sealed selection')
    rows = data['samples']
    aligned_samples({'result': rows}, total)
    boxes = torch.tensor([row['pred_xywh'] for row in rows], dtype=torch.float32)
    targets = torch.tensor([row['target_xywh'] for row in rows], dtype=torch.float32)
    recorded = torch.tensor([row['iou'] for row in rows], dtype=torch.float32)
    computed = bbox_iou(xywh2xyxy(boxes).clamp(0, 1), xywh2xyxy(targets))
    error = (recorded-computed).abs().max().item()
    if not torch.isfinite(computed).all() or error > 1e-6:
        raise ValueError(f'Saved IoUs did not reproduce on CPU: {error}')
    for threshold in (.5, .7):
        if not torch.equal(recorded >= threshold, computed >= threshold):
            raise ValueError('CPU recomputation changed a threshold decision')
    summary = summarize(rows)
    for key in ('total', 'correct_05', 'correct_07', 'acc_05', 'acc_07', 'miou'):
        if abs(summary[key]-data['summary'][key]) > 1e-12:
            raise ValueError('Saved summary differs from per-expression results')
    if split == 'val':
        for current, key in (('acc_05','accu'), ('acc_07','accu_07'), ('miou','miou')):
            tolerance = 1e-5 if key == 'miou' else 1e-12
            if abs(summary[current]-selected['validation'][key]) > tolerance:
                raise ValueError('Selected validation metrics did not reproduce')
    return {'passed': True, 'cpu_iou_max_error': error, 'total': total}


def build_report(root, output, plan):
    import torch
    annotations = torch.load(plan['annotations']['test']['path'], map_location='cpu', weights_only=False)
    evaluations = {name: {split: json.loads((output/name/f'{split}.json').read_text())
                          for split in ('val','test')} for name in plan['checkpoints']}
    verification = {}
    for split, total in (('val',608), ('test',2104)):
        aligned_samples({name:data[split]['samples'] for name,data in evaluations.items()}, total)
        reference_targets = None
        for name,data in evaluations.items():
            verification[f'{name}/{split}'] = verify_result(data[split], plan['checkpoints'][name], split, total)
            targets = [row['target_xywh'] for row in data[split]['samples']]
            if reference_targets is None:
                reference_targets = targets
            elif targets != reference_targets:
                raise ValueError('Same-resolution models received different GT coordinates')
    results = {}
    for name, data in evaluations.items():
        rows = data['test']['samples']
        results[name] = {'val': summarize(data['val']['samples']), 'test': summarize(rows),
            'groups': {'small': summarize([r for r,a in zip(rows,annotations) if str(a[5]).lower()=='small']),
                       'normal': summarize([r for r,a in zip(rows,annotations) if str(a[5]).lower()!='small']),
                       'weak': summarize([r for r,a in zip(rows,annotations) if str(a[4]) in ('1','2')])}}
    paired = {}
    for seed in plan['seeds']:
        control = evaluations[f'lavs_control_seed{seed}']['test']['samples']
        corrected = evaluations[f'corrected_attention_seed{seed}']['test']['samples']
        differential = evaluations[f'differential_attention_seed{seed}']['test']['samples']
        paired[f'corrected_vs_lavs_seed{seed}'] = paired_changes(corrected, control)
        paired[f'differential_vs_lavs_seed{seed}'] = paired_changes(differential, control)
        paired[f'differential_vs_corrected_seed{seed}'] = paired_changes(differential, corrected)
    report = {'kind': 'user_requested_diagnostic_test', 'training_root': str(root),
              'evaluation_plan_sha256': sha256(output/'evaluation_plan.json'),
              'seeds': plan['seeds'], 'multi_seed_replication': len(plan['seeds'])==3,
              'test_used_for_selection': False, 'original_training_gate_unchanged': True,
              'runs': results, 'paired_changes': paired, 'target_acc05': .74,
              'required_hits_of_2104': 1557,
              'target_reached': {name: result['test']['acc_05']>=.74 for name,result in results.items()}}
    original_report = root/'report.json'
    if original_report.exists():
        report['training_promotion'] = json.loads(original_report.read_text()).get('promotion')
    verify_plan(root, output, plan)
    write_json(output/'verification.json', {'passed': True, 'results': verification,
               'original_and_frozen_checkpoints_unchanged': True})
    write_json(output/'report.json', report)
    lines = ['# Differential LAVS diagnostic test', '',
             'All training completed and validation selections sealed before these diagnostic tests.', '',
             '| Model | Test Acc05 | Test Acc07 | Test mIoU | Hits05 / 2104 |',
             '| --- | ---: | ---: | ---: | ---: |']
    for name,result in results.items():
        s = result['test']
        lines.append(f"| {name} | {s['acc_05']*100:.3f}% | {s['acc_07']*100:.3f}% | {s['miou']*100:.3f}% | {s['correct_05']} |")
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    return report


def run(root, output, gpus='auto', wait_seconds=259200):
    output.mkdir(parents=True, exist_ok=True)
    lock = (output/'evaluation.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    sys.path.insert(0, str(root/'source'))
    from script_train.clip_differential_control import runtime as r
    from script_train.clip_differential_control.control import verify_snapshot, gpu_pair
    phase1 = [f'{v}_seed13' for v in r.VARIANTS]
    completed = []
    def status(**values):
        write_json(output/'status.json', {'pid': os.getpid(), 'updated_at': time.time(),
                   'completed': list(completed), 'test_evaluated': False, 'error': None, **values})
    try:
        request = verify_request(root, output)
        deadline = time.monotonic()+wait_seconds
        while True:
            training = json.loads((root/'status.json').read_text())
            if training_ready(root, phase1, r.EPOCHS, training):
                break
            if training.get('phase')=='failed' and not training.get('controller_active'):
                raise RuntimeError('Training failed before completing its queue: '+str(training.get('error')))
            if time.monotonic() >= deadline:
                raise TimeoutError('Training did not finish before the registered wait deadline')
            status(phase='waiting_for_training', completed_training_epochs=training.get('completed_epochs'),
                   current_training_job=training.get('current_job'), training_phase=training['phase'])
            time.sleep(15)
        registration = verify_snapshot(root)
        plan = seal_plan(root, output, registration, r, training, request)
        verify_plan(root, output, plan)
        for name,item in plan['checkpoints'].items():
            while True:
                try:
                    selected_gpus = gpu_pair(gpus)
                    break
                except RuntimeError as error:
                    if time.monotonic() >= deadline:
                        raise TimeoutError('No free GPU pair before evaluation deadline') from error
                    status(phase='waiting_for_gpus', current_case=name, waiting_reason=str(error))
                    time.sleep(15)
            destination = output/name
            destination.mkdir(exist_ok=True)
            command = [r.PYTHON, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2',
                       request['code']['evaluator']['path'], '--checkpoint', item['checkpoint'],
                       '--output-root', str(destination), '--splits', 'val', 'test']
            environment = os.environ.copy()
            environment.update(CUDA_VISIBLE_DEVICES=selected_gpus, PYTHONPATH=str(root/'source'),
                OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', USE_TF='0',
                TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1', MPLCONFIGDIR=str(output/'matplotlib'))
            write_json(destination/'command.json', {'argv': command, 'cwd': str(root/'source'), 'gpus': selected_gpus})
            with (destination/'console.log').open('a') as stream:
                process = subprocess.Popen(command, cwd=root/'source', env=environment,
                    stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                while process.poll() is None:
                    status(phase='evaluating', current_case=name, job_pid=process.pid, gpus=selected_gpus)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        pass
            if process.returncode:
                raise RuntimeError(f'{name} evaluation failed; inspect {destination}/console.log')
            completed.append(name)
        report = build_report(root, output, plan)
        status(phase='complete', test_evaluated=True, test_used_for_selection=False)
        print(json.dumps({name:result['test'] for name,result in report['runs'].items()}, indent=2), flush=True)
    except BaseException as error:
        status(phase='failed', error=str(error))
        raise


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpus', default='auto')
    parser.add_argument('--wait-seconds', type=int, default=259200)
    args = parser.parse_args()
    run(args.root.resolve(), args.output.resolve(), args.gpus, args.wait_seconds)
