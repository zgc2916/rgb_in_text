import inspect
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from engine import _sum_loss_terms
from models.center_evidence_grounder import (CenterEvidenceGrounder, assert_scope,
    center_contract, parameter_group, selected_name)
from models.gentle_differential import GentleDifferentialGrounder
from script_train.clip_gentle_differential_control import runtime as gentle_runtime
from test_gentle_differential import TinyBase
from utils.loss_utils import trans_vg_loss
from utils.misc import NestedTensor

torch.set_num_threads(2)


class ToyVL(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, sequence):
        return sequence + self.encoder.layers[5].weight.mean()


class ToyBase(TinyBase):
    """Real gentle scope, toy forward with PEFT-like adapter switching."""
    def __init__(self, args):
        super().__init__(args)
        self.open_lora = True
        self.infmae_local_center = None
        self.clip.text_projection = nn.Linear(512, 512)
        self.vl_transformer = ToyVL(self.vl_transformer.encoder)
        self.fail_after_text = False
        self.return_aux = False

    def switch_adapter(self, active):
        # PEFT reopens the active adapter and closes the other modality.
        for name, parameter in self.clip.named_parameters():
            if '.lora_A.' in name or '.lora_B.' in name:
                parameter.requires_grad_(f'.lora_{active}.' in name)
        if getattr(self.args, 'match_infmae_lora_schedule', False):
            for name, parameter in self.clip.named_parameters():
                parameter.requires_grad_(self._clip_control_trainability[name])

    def forward(self, image, text):
        self.clip.text_projection(text[:, None].expand(-1, 3, -1))
        eos = self.clip.text_projection(text)
        if self.fail_after_text:
            raise RuntimeError('Synthetic failure after text projection')
        source = image.tensors.mean((1, 2, 3))[:, None].expand(-1, 8)
        image.tensors.add_(2.)  # A misbehaving student must not alter head input.
        projection = self.clip.vision_model.encoder.layers[8].self_attn.q_proj
        features = []
        for modality in ('rgb', 'ir'):
            self.switch_adapter(modality)
            features.append(projection.lora_B[f'lora_{modality}'](
                projection.lora_A[f'lora_{modality}'](source)))
        self.switch_adapter('rgb')
        visual = features[0] + features[1]
        sequence = torch.stack((eos + visual.mean(-1, keepdim=True), eos), dim=0)
        self.vl_transformer(sequence)
        box = self.bbox_embed(self.vl_transformer.encoder.layers[5](visual)).sigmoid()
        main = (box, eos[:, :2], [eos[:, :2], eos[:, :2]], eos[:, :4], eos[:, :4])
        return (*main, {'student_aux': box.mean()}) if self.return_aux else main


def create(tmp_path, enabled=True, mode='text_gate', freeze=False):
    args = gentle_runtime.recipe(tmp_path, 'gentle_differential', 13)
    args.center_evidence_enabled = enabled
    args.center_evidence_mode = mode
    args.center_evidence_freeze_student = freeze
    args.center_evidence_loss_weight = .05
    args.imsize = 32
    args.center_evidence_hidden = 16
    args.center_evidence_min_window_pixels = 16.
    args.center_evidence_min_shift_pixels = 4.
    args.center_evidence_max_shift_pixels = 8.
    student = GentleDifferentialGrounder(ToyBase(args), args)
    return CenterEvidenceGrounder(student), args


def inputs():
    generator = torch.Generator().manual_seed(222)
    images = torch.randn(2, 4, 32, 32, generator=generator)
    padding = torch.zeros(2, 32, 32, dtype=torch.bool)
    padding[0, :, 28:] = True
    text = torch.randn(2, 512, generator=generator)
    return NestedTensor(images, padding), text


def loss_args(**kwargs):
    return SimpleNamespace(use_contrastive_loss=False, use_rtcc_constrain_loss=False,
                           use_mask_loss=False, **kwargs)


def test_default_disabled_delegates_unchanged_and_loss_ignores_aux(tmp_path):
    model, args = create(tmp_path, enabled=False)
    model.eval()
    img, text = inputs()
    original = img.tensors.clone()
    with torch.no_grad():
        baseline = model.student(NestedTensor(img.tensors.clone(), img.mask.clone()), text)
        output = model(img, text)
    assert model.center_head is None and len(output) == len(baseline) == 5
    assert torch.equal(output[0], baseline[0]) and torch.equal(img.tensors, original)
    assert not any(name.startswith('center_head.') for name, _ in model.named_parameters())
    target = output[0].clone(); target[:, 0] += .03
    plain = trans_vg_loss(args, output[0], target, None, None)
    ignored = trans_vg_loss(args, output[0], target, None, None,
                            aux={'center_evidence': object()})
    assert plain.keys() == ignored.keys()
    assert all(torch.equal(plain[key], ignored[key]) for key in plain)


@pytest.mark.parametrize('mode', ['fixed', 'text_gate'])
def test_enabled_head_is_exact_student_at_zero_init_and_keeps_five_eval_outputs(tmp_path, mode):
    model, _ = create(tmp_path, mode=mode)
    model.eval()
    img, text = inputs(); original = img.tensors.clone()
    with torch.no_grad():
        baseline = model.student(NestedTensor(img.tensors.clone(), img.mask.clone()), text)
        output = model(img, text)
    assert len(output) == 5 and torch.equal(output[0], baseline[0])
    assert torch.equal(img.tensors, original)
    assert 'target_boxes' not in inspect.signature(model.forward).parameters
    with pytest.raises(TypeError):model(img, text, target_boxes=baseline[0])


@pytest.mark.parametrize('freeze', [False, True])
def test_peft_adapter_switches_keep_both_scopes_and_dropout_policy(tmp_path, freeze):
    model, args = create(tmp_path, mode='text_gate', freeze=freeze)
    model.train()
    layers = model.student.base.clip.vision_model.encoder.layers
    assert model.training and model.center_head.training
    assert layers[8].self_attn.q_proj.lora_dropout['lora_rgb'].training is (not freeze)
    assert layers[11].self_attn.q_proj.lora_dropout['lora_ir'].training is (not freeze)
    assert not layers[7].self_attn.q_proj.lora_dropout['lora_rgb'].training
    out = model(*inputs())
    assert len(out) == 6 and 'center_evidence' in out[5]
    assert_scope(model, args)
    for modality in ('rgb','ir'):
        p = layers[8].self_attn.q_proj.lora_A[f'lora_{modality}'].weight
        assert p.requires_grad is (not freeze)
    assert model.student.base._clip_control_trainability == {
        n:p.requires_grad for n,p in model.student.base.clip.named_parameters()}
    model.eval()
    assert len(model(*inputs())) == 5
    assert not layers[8].self_attn.q_proj.lora_dropout['lora_rgb'].training


def test_frozen_student_keeps_deterministic_dropout_during_train(tmp_path):
    model, args = create(tmp_path, freeze=True)
    model.train()
    assert not model.student.training and not model.student.base.training
    assert all(not p.requires_grad for p in model.student.parameters())
    assert all(not value for value in model.student.base._clip_control_trainability.values())
    assert_scope(model, args)


@pytest.mark.parametrize('mode', ['fixed', 'text_gate'])
def test_parameter_group_contract_respects_mode_and_freeze(tmp_path, mode):
    model, args = create(tmp_path, mode=mode)
    for name, parameter in model.named_parameters():
        assert parameter.requires_grad == selected_name(name, args)
    assert parameter_group('student.base.bbox_embed.weight', args) == 'decoder'
    assert parameter_group('student.base.clip.text_projection.weight', args) is None
    assert parameter_group('center_head.offset.2.weight', args) == 'center'
    assert (parameter_group('center_head.route.2.weight', args) == 'center') is (mode == 'text_gate')
    frozen, frozen_args = create(tmp_path/'frozen', mode=mode, freeze=True)
    assert all(not selected_name('student.'+name, frozen_args) and not parameter.requires_grad
               for name, parameter in frozen.student.named_parameters())
    assert_scope(frozen, frozen_args)


def test_loss_dispatch_adds_real_heatmap_and_backpropagates_through_adapters(tmp_path):
    model, args = create(tmp_path, mode='text_gate')
    model.train()
    img, text = inputs()
    result = model(img, text)
    main, aux = result[:5], result[5]
    target = main[0].detach().clone(); target[:, 0] += .01
    losses = trans_vg_loss(args, main[0], target, None, None, aux=aux)
    assert losses['loss_center_heatmap'] > 0
    assert losses['center_eligible_fraction'] > 0 and not losses['center_eligible_fraction'].requires_grad
    _sum_loss_terms(losses).backward()
    assert model.center_head.rgb_projection.weight.grad.abs().sum() > 0
    assert model.center_head.tir_projection.weight.grad.abs().sum() > 0
    assert model.center_head.route[-1].weight.grad.abs().sum() > 0
    layers = model.student.base.clip.vision_model.encoder.layers
    for modality in ('rgb','ir'):
        grad = layers[8].self_attn.q_proj.lora_A[f'lora_{modality}'].weight.grad
        assert grad is not None and grad.abs().sum() > 0
    assert_scope(model,args)


def test_enabled_loss_rejects_missing_aux_and_invalid_weight():
    pred = torch.tensor([[.5,.5,.2,.2]], requires_grad=True)
    target = torch.tensor([[.5,.5,.2,.2]])
    args = loss_args(center_evidence_enabled=True)
    with pytest.raises(ValueError, match='auxiliary'):
        trans_vg_loss(args,pred,target,None,None)
    args.center_evidence_loss_weight = float('nan')
    with pytest.raises(ValueError, match='finite and non-negative'):
        trans_vg_loss(args,pred,target,None,None,aux={'center_evidence':{}})


def test_hooks_are_removed_after_student_and_registration_failures(tmp_path, monkeypatch):
    model, _ = create(tmp_path)
    model.train()
    base = model.student.base
    initial = (len(base.clip.text_projection._forward_hooks),len(base.vl_transformer._forward_hooks))
    base.fail_after_text = True
    with pytest.raises(RuntimeError, match='Synthetic failure'):
        model(*inputs())
    assert (len(base.clip.text_projection._forward_hooks),len(base.vl_transformer._forward_hooks)) == initial
    base.fail_after_text = False
    original = base.vl_transformer.register_forward_hook
    def fail_registration(*args,**kwargs):raise RuntimeError('Hook registration failure')
    monkeypatch.setattr(base.vl_transformer,'register_forward_hook',fail_registration)
    with pytest.raises(RuntimeError, match='Hook registration failure'):model(*inputs())
    assert (len(base.clip.text_projection._forward_hooks),len(base.vl_transformer._forward_hooks)) == initial
    monkeypatch.setattr(base.vl_transformer,'register_forward_hook',original)
    assert len(model(*inputs())) == 6
    assert (len(base.clip.text_projection._forward_hooks),len(base.vl_transformer._forward_hooks)) == initial


def test_student_aux_is_preserved_and_configuration_mutation_rejected(tmp_path):
    model, args = create(tmp_path)
    model.student.base.return_aux = True
    model.train()
    out = model(*inputs())
    assert set(out[5]) == {'student_aux','center_evidence'}
    args.center_evidence_mode = 'fixed'
    with pytest.raises(ValueError, match='configuration changed'):
        model(*inputs())


def test_head_construction_does_not_consume_global_rng(tmp_path):
    args = gentle_runtime.recipe(tmp_path,'gentle_differential',13)
    args.center_evidence_enabled = True
    args.center_evidence_mode = 'fixed'
    args.imsize = 32
    args.center_evidence_min_window_pixels = 16.
    args.center_evidence_min_shift_pixels = 4.
    args.center_evidence_max_shift_pixels = 8.
    student = GentleDifferentialGrounder(ToyBase(args),args)
    before = torch.get_rng_state().clone()
    CenterEvidenceGrounder(student)
    assert torch.equal(torch.get_rng_state(),before)


def test_unrestored_peft_configuration_is_rejected_when_enabled(tmp_path):
    args = gentle_runtime.recipe(tmp_path,'gentle_differential',13)
    args.center_evidence_enabled = True
    args.match_infmae_lora_schedule = False
    student = GentleDifferentialGrounder(ToyBase(args),args)
    with pytest.raises(ValueError,match='PEFT adapter restore mask'):
        CenterEvidenceGrounder(student)
    args.center_evidence_enabled = False
    assert not center_contract(args)['enabled']
