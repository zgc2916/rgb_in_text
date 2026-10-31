"""Validation-only, seed-matched evidence for the bounded warm-start suite."""
import json
from pathlib import Path
import statistics

from .runtime import REFERENCE, sha256
from script_train.clip_structure.report import metrics, image_paired_bootstrap


def replication_check(runs,baselines):
    keys=('accu','accu_07','miou','miou_weak')
    if set(runs)!={13,14,15}:
        return {'attempted':False,'passed':False}
    deltas={str(seed):{key:runs[seed][key]-baselines[seed][key] for key in keys} for seed in (13,14,15)}
    means={key:statistics.mean(deltas[str(seed)][key] for seed in (13,14,15)) for key in keys}
    changes=[deltas[str(seed)]['accu'] for seed in (13,14,15)]
    passed=(sum(value>1e-12 for value in changes)>=2 and all(value>=-1e-12 for value in changes)
            and means['accu']>=.005-1e-12 and all(means[key]>=-1e-12 for key in keys[1:]))
    return {'attempted':True,'passed':passed,'paired_deltas':deltas,'mean_deltas':means,
            'mean_metrics':{key:statistics.mean(runs[seed][key] for seed in (13,14,15)) for key in keys},
            'sample_sd':{key:statistics.stdev(runs[seed][key] for seed in (13,14,15)) for key in keys}}


def report(root):
    root=Path(root)
    manifest=json.loads((root/'preregistration.json').read_text())
    selected=json.loads((root/'selected_models.json').read_text())
    promotion=json.loads((root/'promotion.json').read_text())
    references={item['seed']:item for item in manifest['references']}
    result={'stage':'frozen-trunk warm-start exploration, not a common-initialization formal comparison',
        'selection':manifest['selection'],'promotion':promotion,'runs':{},'mechanism_comparisons':{},
        'automatic_test_evaluation':False,'test_previously_observed':True,
        'novelty_status':'hypothesis requiring additional comparisons and literature review'}
    predictions={}
    for item in selected:
        if sha256(item['checkpoint'])!=item['sha256']:
            raise ValueError('Frozen candidate changed')
        evaluation=json.loads((root/'evaluation'/item['name']/'val.json').read_text())
        if evaluation['source']['sha256']!=item['sha256'] or evaluation['source']['epoch']!=item['epoch']:
            raise ValueError('Evaluation did not use the selected checkpoint')
        expected=item['validation']
        actual=evaluation['summary']
        for new,old in [('acc_05','accu'),('acc_07','accu_07'),('miou','miou')]:
            if abs(actual[new]-expected[old])>1e-7:
                raise ValueError(f'Frozen validation differs from selection: {item["name"]}, {new}')
        baseline=json.loads((REFERENCE/'evaluation'/f"c1_seed{item['seed']}"/'val.json').read_text())['samples']
        rows=evaluation['samples']
        predictions[item['name']]=rows
        result['runs'][item['name']]={'epoch':item['epoch'],'base_sha256':item['base_sha256'],
            'validation':metrics(rows),'validation_groups':expected,
            'paired_c1':image_paired_bootstrap(rows,baseline)}
    for label,candidate,control in [('reliability_vs_direction','reliability_gate','direction_gate'),
                                   ('spatial_vs_reliability','reliability_spatial','reliability_gate')]:
        result['mechanism_comparisons'][label]=image_paired_bootstrap(
            predictions[candidate+'_seed13'],predictions[control+'_seed13'])
    winner=promotion['selected_variant']
    runs={item['seed']:item['validation'] for item in selected if item['variant']==winner}
    result['replication']=replication_check(runs,{seed:item['validation'] for seed,item in references.items()})
    result['recommended_model']=winner if result['replication']['passed'] else 'c1'
    result['interpretation']='Validation-based exploration only; no claim of guaranteed test improvement. Initial C1 remains available.'
    return result
