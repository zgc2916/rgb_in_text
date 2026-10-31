import json
from pathlib import Path
import statistics

from .runtime import REFERENCE,REPOSITORY,sha256
from script_train.clip_structure.report import metrics,image_paired_bootstrap

KEYS=('accu','accu_07','miou','miou_weak')


def check_replication(candidates,baselines,minimum):
    if set(candidates)!={13,14,15}:
        return {'attempted':False,'passed':False}
    deltas={str(seed):{key:candidates[seed][key]-baselines[seed][key] for key in KEYS} for seed in (13,14,15)}
    mean={key:statistics.mean(deltas[str(seed)][key] for seed in (13,14,15)) for key in KEYS}
    changes=[deltas[str(seed)]['accu'] for seed in (13,14,15)]
    passed=(sum(value>1e-12 for value in changes)>=2 and all(value>=-1e-12 for value in changes)
        and mean['accu']>=minimum-1e-12 and all(mean[key]>=-1e-7 for key in KEYS[1:]))
    return {'attempted':True,'passed':passed,'paired_deltas':deltas,'mean_deltas':mean,
        'mean_metrics':{key:statistics.mean(candidates[seed][key] for seed in (13,14,15)) for key in KEYS},
        'sample_sd':{key:statistics.stdev(candidates[seed][key] for seed in (13,14,15)) for key in KEYS}}


def report(root):
    root=Path(root)
    manifest=json.loads((root/'preregistration.json').read_text())
    selected=json.loads((root/'selected_models.json').read_text())
    promotion=json.loads((root/'promotion.json').read_text())
    result={'stage':manifest['stage'],'selection':manifest['selection'],'promotion':promotion,
        'test_used':False,'test_previously_observed':True,'runs':{},'mechanism_comparisons':{},
        'formal_common_initialization_comparison_completed':False}
    historical=REPOSITORY/'run_resources/augmented_direction_log.txt'
    if historical.exists():
        history=[json.loads(line) for line in historical.read_text().splitlines() if line.strip()]
        result['historical_augmented_control']={'case':'direction gate, probability0.35, original losses',
            'epochs':len(history),'acc05_trajectory':[row['val_accu'] for row in history],
            'source':'immutable source/run_resources/augmented_direction_log.txt'}
    rows_by_name={}
    for item in selected:
        if sha256(item['checkpoint'])!=item['sha256']:
            raise ValueError('Frozen checkpoint changed')
        evaluation=json.loads((root/'evaluation'/item['name']/'val.json').read_text())
        if evaluation['source']['sha256']!=item['sha256'] or evaluation['source']['epoch']!=item['epoch']:
            raise ValueError('Evaluation did not use the selected checkpoint')
        for a,b in [('acc_05','accu'),('acc_07','accu_07'),('miou','miou')]:
            if abs(evaluation['summary'][a]-item['validation'][b])>1e-7:
                raise ValueError('Reconstructed validation differs from training selection')
        rows=evaluation['samples']
        baseline=json.loads((REFERENCE/'evaluation'/f"c1_seed{item['seed']}"/'val.json').read_text())['samples']
        rows_by_name[item['name']]=rows
        log=[json.loads(line) for line in (root/'explore'/item['name']/'log.txt').read_text().splitlines() if line.strip()]
        trained=max(log,key=lambda row:row['val_accu'])
        result['runs'][item['name']]={'epoch':item['epoch'],'selected_initial_c1':item['epoch']==-1,
            'validation':metrics(rows),'validation_groups':item['validation'],
            'best_trained_epoch':trained['epoch'],
            'best_trained_validation':{key:trained['val_'+key] for key in KEYS},
            'acc05_trajectory':[row['val_accu'] for row in log],
            'paired_c1':image_paired_bootstrap(rows,baseline)}
    for name,candidate,control in [('box_vs_full_loss','b_clean_box','a_clean_full'),
            ('joint_vs_frozen','c_joint_box','b_clean_box'),('fusion_vs_continued_c1','c_joint_box','d_c1_continue')]:
        result['mechanism_comparisons'][name]=image_paired_bootstrap(
            rows_by_name[candidate+'_seed13'],rows_by_name[control+'_seed13'])
    baselines={item['seed']:item['validation'] for item in manifest['references']}
    controls={item['seed']:item['validation'] for item in selected if item['variant']=='d_c1_continue'}
    joint={item['seed']:item['validation'] for item in selected if item['variant']=='c_joint_box'}
    control_check=check_replication(controls,baselines,.005)
    joint_check=check_replication(joint,baselines,.005)
    paired_check=check_replication(joint,controls,.003) if set(controls)=={13,14,15} else {'attempted':False,'passed':False}
    fusion_pass=joint_check['passed'] and paired_check['passed']
    result['replication']={'passed':fusion_pass or control_check['passed'],
        'joint_vs_original':joint_check,'joint_vs_continued':paired_check,'continued_vs_original':control_check}
    result['recommended_model']='c_joint_box' if fusion_pass else ('d_c1_continue' if control_check['passed'] else 'c1')
    result['fusion_contribution_supported_by_replication']=fusion_pass
    result['interpretation']='Validation-selected controlled continuation. A training-only gain does not establish a fusion contribution; no guaranteed test improvement.'
    return result
