"""Complete reporting after the 30-vs-20 epoch audit configuration error."""
import argparse
import json
from pathlib import Path
import sys
import time


def recover(root):
    # The experiment snapshot stays immutable. Only report generation uses
    # the explicitly registered epoch count instead of the inherited default.
    sys.path.insert(0, str(root / 'source'))
    from script_train.clip_scale_geometry_control import runtime as r
    from script_train.clip_scale_geometry_control.control import verify_snapshot
    from script_train.clip_reliability_control.runtime import best_from_log
    import torch

    state = json.loads((root / 'status.json').read_text())
    if (state['phase'] != 'failed' or state.get('controller_active') or state.get('job_pid')
            or not str(state.get('error', '')).startswith('Incomplete exploration:')):
        raise ValueError('This is not the inactive, report-only epoch-count failure')
    manifest = verify_snapshot(root)
    if manifest['epochs_per_run'] != r.EPOCHS:
        raise ValueError('Registered and frozen runtime epoch counts differ')
    protected = {}
    for variant in r.VARIANTS:
        name = f'{variant}_seed13'
        job = state['jobs'][name]
        if not job['completed'] or job['exit_code'] != 0:
            raise ValueError('A training job did not complete successfully')
        directory = root / 'explore' / name
        progress = json.loads((directory / 'progress.json').read_text())
        if progress['completed_epochs'] != r.EPOCHS or progress['epoch_in_progress']:
            raise ValueError('A registered epoch is missing')
        for stem in ('best_checkpoint.pth', 'trained_best_checkpoint.pth', 'checkpoint.pth'):
            path = directory / stem
            protected[str(path)] = r.sha256(path)
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            if checkpoint['provenance'] != manifest['provenance']:
                raise ValueError('Checkpoint provenance differs')
            expected_epoch = (r.EPOCHS - 1 if stem == 'checkpoint.pth' else
                json.loads((directory / ('best.json' if stem == 'best_checkpoint.pth' else 'trained_best.json')).read_text())['epoch'])
            if checkpoint['epoch'] != expected_epoch:
                raise ValueError('Checkpoint and metadata epoch differ')
            del checkpoint
    runs = {v: best_from_log(root / 'explore' / f'{v}_seed13', epochs=r.EPOCHS) for v in r.VARIANTS}
    promotion = r.choose_promotion(manifest['references'][0]['validation'], runs)
    report = {'stage': 'single_seed_warm_start_screening', 'runs': runs,
              'best_trained': {v: json.loads((root / 'explore' / f'{v}_seed13/trained_best.json').read_text())
                               for v in r.VARIANTS},
              'promotion': promotion, 'test_evaluated': False, 'multi_seed_replication_completed': False,
              'recommended_model': 'c1_pending_replication' if promotion['selected_variant'] else 'c1',
              'report_recovery': 'Explicit registered 30-epoch completeness check; training and selection unchanged'}
    for path, digest in protected.items():
        if r.sha256(path) != digest:
            raise ValueError('A model changed during report recovery')
    r.write_json(root / 'report_recovery_2026-10-28.json', {
        'previous_status': state, 'cause': 'best_from_log defaulted to the previous suite\u2019s 20 epochs',
        'registered_epochs_used': r.EPOCHS, 'training_repeated': False, 'selection_changed': False,
        'frozen_training_source_unchanged': True, 'checkpoint_hashes_verified': protected,
        'recovered_at': time.time()})
    r.write_json(root / 'report.json', report)
    state.update(phase='phase1_complete' if promotion['selected_variant'] else 'gate_not_passed',
                 current_job=None, completed_epochs=len(r.VARIANTS)*r.EPOCHS, controller_active=False,
                 error=None, recommended_model=report['recommended_model'], updated_at=time.time(),
                 report_recovery='report_recovery_2026-10-28.json')
    r.write_json(root / 'status.json', state)
    print(json.dumps({'phase': state['phase'], 'completed_epochs': state['completed_epochs'],
                      'promotion': promotion}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('output_clip_lavs_control/2026-10-28_scale_geometry'))
    recover(parser.parse_args().root.resolve())
