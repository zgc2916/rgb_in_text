import copy

import pytest
import torch
from torch import nn

from models.differential_lavs import DifferentialLanguageBlock, DifferentialVisualExchange
from models.gentle_differential import GentleDifferentialGrounder, parameter_group
from script_train.clip_gentle_differential_control import runtime as r


class TinyBase(nn.Module):
    """Exercise the real replacement blocks and representative C1 scopes."""
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.clip = nn.Module()
        self.clip.text_model = nn.Linear(8, 8)
        self.clip.vision_model = nn.Module()
        self.clip.vision_model.encoder = nn.Module()
        self.clip.vision_model.encoder.layers = nn.ModuleList()
        for i in range(12):
            layer = nn.Module()
            layer.self_attn = nn.Module()
            layer.self_attn.q_proj = nn.Module()
            layer.self_attn.q_proj.base_layer = nn.Linear(8, 8)
            layer.self_attn.q_proj.lora_A = nn.ModuleDict({name: nn.Linear(8, 2, bias=False) for name in ('lora_rgb', 'lora_ir')})
            layer.self_attn.q_proj.lora_B = nn.ModuleDict({name: nn.Linear(2, 8, bias=False) for name in ('lora_rgb', 'lora_ir')})
            layer.self_attn.q_proj.lora_dropout = nn.ModuleDict({name: nn.Dropout(.1) for name in ('lora_rgb', 'lora_ir')})
            if i in (0, 3, 7, 11):
                layer.language_replacement = DifferentialLanguageBlock(8, 2, 16, enabled=args.differential_enabled)
            self.clip.vision_model.encoder.layers.append(layer)
        self.differential_exchange = DifferentialVisualExchange(8, 2, levels=2, enabled=args.differential_enabled)
        self.cross_fusion_layers_vt = nn.ModuleList()
        self.cross_fusion_layers_tv = nn.ModuleList()
        self.vl_transformer = nn.Module()
        self.vl_transformer.encoder = nn.Module()
        self.vl_transformer.encoder.layers = nn.ModuleList([nn.Linear(8, 8) for _ in range(6)])
        self.vl_transformer.encoder.norm = nn.LayerNorm(8)
        self.bbox_embed = nn.Linear(8, 4)
        self.reg_token = nn.Embedding(1, 8)
        self.iafv3_token_gate = nn.Linear(8, 1)
        self.visu_proj = nn.Linear(8, 8)
        self.text_proj = nn.Linear(8, 8)
        self.ml_visual_projection = nn.Linear(8, 8)
        self.visu_token_norm = nn.LayerNorm(8)
        self.visu_token_mlp = nn.Linear(8, 8)
        self.tir_detail = nn.Linear(8, 8)


def create(tmp_path, case='gentle_differential'):
    args = r.recipe(tmp_path, case, 13)
    return GentleDifferentialGrounder(TinyBase(args), args), args


def validation(accu=.78, small=.65):
    return {'accu': accu, 'accu_07': .58, 'miou': .66, 'accu_small': small,
            'miou_weak': .63, 'accu_normal': .97}


def test_recipes_are_paired_and_preserve_the_requested_training_budget(tmp_path):
    corrected = r.recipe(tmp_path/'corrected', 'gentle_corrected', 13)
    differential = r.recipe(tmp_path/'differential', 'gentle_differential', 13)
    assert corrected.replace_lavs and differential.replace_lavs
    assert corrected.visual_exchange == differential.visual_exchange == 'differential_lavs'
    assert not corrected.differential_enabled and differential.differential_enabled
    assert corrected.exchange_base_checkpoint == differential.exchange_base_checkpoint
    assert corrected.exchange_base_sha256 == differential.exchange_base_sha256
    for args in (corrected, differential):
        assert args.imsize == 224 and args.epochs == 60
        assert args.batch_size * args.grad_accum_steps * args.world_size == 96
        assert args.projection_lr == 2e-6 and args.late_visual_lora_lr == 4e-6
        assert args.replacement_lr == 2e-5 and args.ema_decay == .99
        assert args.teacher_preserve_epochs == 0 and args.teacher_preserve_weight == 0.
        assert args.clip_optimizer_groups_separately
    assert r.ORDER[0] == 'gentle_differential'
    with pytest.raises(ValueError, match='Unregistered'):
        r.recipe(tmp_path, 'other', 13)
    with pytest.raises(ValueError, match='Unregistered'):
        r.recipe(tmp_path, 'gentle_corrected', 42)


@pytest.mark.parametrize('case,enabled', [('gentle_corrected', False), ('gentle_differential', True)])
def test_tuning_scope_matches_late_adaptation_and_optionally_reference_attention(tmp_path, case, enabled):
    model, args = create(tmp_path, case)
    parameters = dict(model.named_parameters())
    def trained(name):
        return parameters[name].requires_grad
    # Same last-four visual LoRA policy as the prior late-adaptation model.
    for modality in ('lora_rgb', 'lora_ir'):
        assert trained(f'base.clip.vision_model.encoder.layers.8.self_attn.q_proj.lora_A.{modality}.weight')
        assert trained(f'base.clip.vision_model.encoder.layers.11.self_attn.q_proj.lora_B.{modality}.weight')
        assert not trained(f'base.clip.vision_model.encoder.layers.7.self_attn.q_proj.lora_A.{modality}.weight')
    assert not trained('base.clip.vision_model.encoder.layers.8.self_attn.q_proj.base_layer.weight')
    assert not trained('base.clip.text_model.weight')
    # All migrated encoder-internal foreground stays fixed; only the new
    # reference is allowed to update in the actual differential variant.
    inner = 'base.clip.vision_model.encoder.layers.3.language_replacement.branches.rgb.'
    assert not trained(inner+'norm.weight')
    assert not trained(inner+'attention.query.weight')
    assert not trained(inner+'expand.weight')
    assert trained(inner+'attention.head_coefficient') == enabled
    assert trained(inner+'attention.reference_query.weight') == enabled
    # External migrated foreground is the replacement for trainable LAVS.
    assert trained('base.differential_exchange.rgb.0.query.weight')
    assert trained('base.differential_exchange.ir.1.output.weight')
    assert trained('base.differential_exchange.rgb.0.reference_text.weight') == enabled
    assert not trained('base.vl_transformer.encoder.layers.2.weight')
    assert trained('base.vl_transformer.encoder.layers.3.weight')
    assert trained('base.vl_transformer.encoder.layers.5.weight')
    for name in ('base.vl_transformer.encoder.norm.weight', 'base.iafv3_token_gate.weight',
                 'base.visu_proj.weight', 'base.text_proj.weight', 'base.ml_visual_projection.weight',
                 'base.bbox_embed.weight', 'base.reg_token.weight'):
        assert trained(name)
    assert not trained('base.tir_detail.weight')
    assert model.base._clip_control_trainability == {n:p.requires_grad for n,p in model.base.clip.named_parameters()}
    r.assert_scope(model, args)
    optimizer, _ = r.optimizer_for(model, args)
    lrs = {group['name']:group['initial_lr'] for group in optimizer.param_groups}
    assert lrs['decoder'] == lrs['fusion'] == 2e-6 and lrs['visual_lora'] == 4e-6
    assert ('differential' in lrs) == enabled
    if enabled:
        assert lrs['differential'] == 2e-5
    assert {id(p) for group in optimizer.param_groups for p in group['params']} == {
        id(p) for p in model.parameters() if p.requires_grad}


def test_train_mode_matches_late_lora_dropout_policy_without_unfreezing_base(tmp_path):
    model,args = create(tmp_path)
    model.train()
    layers = model.base.clip.vision_model.encoder.layers
    assert not model.base.training and not model.base.clip.training
    assert not layers[7].self_attn.q_proj.lora_dropout['lora_rgb'].training
    assert layers[8].self_attn.q_proj.lora_dropout['lora_rgb'].training
    assert layers[11].self_attn.q_proj.lora_dropout['lora_ir'].training
    r.assert_scope(model,args)
    model.eval()
    assert not layers[8].self_attn.q_proj.lora_dropout['lora_rgb'].training


def test_zero_reference_matches_corrected_and_real_updates_reach_reference(tmp_path):
    torch.manual_seed(20)
    corrected,_ = create(tmp_path/'control', 'gentle_corrected')
    torch.manual_seed(20)
    model,args = create(tmp_path/'candidate', 'gentle_differential')
    control = corrected.base.differential_exchange.rgb[0]
    block = model.base.differential_exchange.rgb[0]
    query,source,words = torch.randn(2,4,8),torch.randn(2,6,8),torch.randn(2,5,512)
    assert torch.equal(control(query,source,context=words),block(query,source,context=words))
    optimizer,_ = r.optimizer_for(model,args)
    before = block.reference_query.weight.detach().clone()
    for _ in range(2):
        optimizer.zero_grad()
        block(query,source,context=words).square().mean().backward()
        optimizer.step()
    assert block.head_coefficient.grad.abs().sum() > 0
    assert block.coefficient_route.weight.grad.abs().sum() > 0
    assert block.reference_query.weight.grad.abs().sum() > 0
    assert not torch.equal(before, block.reference_query.weight)
    assert not torch.equal(control(query,source,context=words),block(query,source,context=words))


def test_delta_restores_exactly_and_rejects_wrong_schema_before_mutation(tmp_path):
    model,args = create(tmp_path)
    record = r.delta_record(model,args,2,validation(),.78,{'source':'preflight'})
    saved = {name:tensor.clone() for name,tensor in record['model_delta'].items()}
    with torch.no_grad():
        for name,p in model.named_parameters():
            if r.selected_name(name,args):p.add_(1.)
    r.restore_delta(model,args,record)
    assert all(torch.equal(model.state_dict()[name],tensor) for name,tensor in saved.items())
    wrong = copy.deepcopy(record)
    names = list(wrong['model_delta'])
    wrong['model_delta'][names[0]].add_(2.)
    wrong['model_delta'][names[-1]] = wrong['model_delta'][names[-1]].double()
    with pytest.raises(ValueError, match='schema'):
        r.restore_delta(model,args,wrong)
    assert all(torch.equal(model.state_dict()[name],tensor) for name,tensor in saved.items())
    wrong = copy.deepcopy(record)
    wrong['model_delta'].pop(next(iter(wrong['model_delta'])))
    with pytest.raises(ValueError, match='scope'):
        r.restore_delta(model,args,wrong)
    wrong = copy.deepcopy(record)
    wrong['model_delta'][names[0]].add_(2.)
    model.base.clip.text_model.weight.requires_grad_(True)
    with pytest.raises(AssertionError, match='tuning mask'):
        r.restore_delta(model,args,wrong)
    assert all(torch.equal(model.state_dict()[name],tensor) for name,tensor in saved.items())


def test_variant_and_seed_mismatches_cannot_build_or_restore_silently(tmp_path):
    model,args = create(tmp_path/'corrected','gentle_corrected')
    record = r.delta_record(model,args,0,validation(),.78,{})
    other = r.recipe(tmp_path/'differential','gentle_differential',13)
    with pytest.raises(ValueError,match='different gentle experiment'):
        r.restore_delta(model,other,record)
    wrong = copy.deepcopy(record)
    wrong['experiment_case'] = 'gentle_differential'
    with pytest.raises(ValueError,match='different experiment'):
        r.restore_delta(model,args,wrong)
    wrong = copy.deepcopy(record)
    wrong['args'].seed = 14
    with pytest.raises(ValueError,match='different experiment'):
        r.restore_delta(model,args,wrong)
    # Even synchronized mutations to the shared model args cannot relabel an
    # already constructed corrected model as differential.
    args.experiment_case = 'gentle_differential'
    args.differential_enabled = True
    with pytest.raises(ValueError,match='different gentle experiment'):
        r.assert_scope(model,args)


def test_recipe_rejects_a_variant_label_that_disagrees_with_structure(tmp_path):
    args = r.recipe(tmp_path,'gentle_corrected')
    args.differential_enabled = True
    with pytest.raises(ValueError,match='disagrees'):
        r.build(args)
    args.differential_enabled = False
    args.visual_exchange = 'lavs'
    with pytest.raises(ValueError,match='disagrees'):
        r.build(args)


def test_changed_module_mode_cannot_disagree_with_registered_variant(tmp_path):
    model,args = create(tmp_path,'gentle_corrected')
    model.base.differential_exchange.rgb[0].enabled = True
    with pytest.raises(ValueError,match='Built attention modules'):
        r.assert_scope(model,args)


def test_validation_gate_requires_a_real_gain_over_both_controls(monkeypatch):
    base = validation(460/608,.60)
    monkeypatch.setattr(r,'reference',lambda seed:{'validation':base})
    corrected = validation(462/608,.61)
    candidate = validation(464/608,.62)
    runs = {'gentle_corrected_seed13':corrected,'gentle_differential_seed13':candidate}
    assert r.choose_promotion(runs)['passed']
    runs['gentle_differential_seed13'] = validation(463/608,.62)
    assert not r.choose_promotion(runs)['passed']
    runs['gentle_differential_seed13'] = dict(candidate,miou_weak=.62)
    assert not r.choose_promotion(runs)['passed']
    runs['gentle_differential_seed13'] = dict(candidate,accu_small=corrected['accu_small'])
    assert not r.choose_promotion(runs)['passed']


def test_replication_gate_requires_matched_multi_seed_improvement(monkeypatch):
    base = validation(460/608,.60)
    monkeypatch.setattr(r,'reference',lambda seed:{'validation':base})
    runs = {}
    for seed in r.SEEDS:
        runs[f'gentle_corrected_seed{seed}'] = validation(461/608,.61)
        runs[f'gentle_differential_seed{seed}'] = validation(464/608,.62)
    assert r.replication_gate(runs,'gentle_differential')['passed']
    for seed in (14,15):
        runs[f'gentle_differential_seed{seed}'] = validation(460/608,.60)
    assert not r.replication_gate(runs,'gentle_differential')['passed']
    with pytest.raises(ValueError,match='registered'):
        r.replication_gate(runs,'gentle_corrected')


def test_training_reuses_common_loop_with_explicit_runtime_injection(monkeypatch,tmp_path):
    seen = {}
    def common_train(root, directory, variant, seed, resume=False, runtime=None):
        seen.update(root=root, directory=directory, variant=variant, seed=seed, resume=resume, runtime=runtime)
        return 'called'
    monkeypatch.setattr(r.common,'train',common_train)
    assert r.train(tmp_path,tmp_path/'run','gentle_differential',13,True) == 'called'
    assert seen == {'root':tmp_path,'directory':tmp_path/'run','variant':'gentle_differential',
                    'seed':13,'resume':True,'runtime':r}
