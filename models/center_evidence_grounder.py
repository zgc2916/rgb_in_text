"""Optional center-evidence wrapper around a gentle differential student.

The forward API receives no ground truth. Training supervision is dispatched
separately by ``utils.loss_utils.trans_vg_loss`` from the sixth auxiliary item.
"""
import torch
from torch import nn

from utils.misc import NestedTensor
from .center_evidence import CenterEvidenceHead
from .gentle_differential import (GentleDifferentialGrounder, assert_model_contract,
                                  parameter_group as gentle_parameter_group,
                                  selected_name as gentle_selected_name)


def center_contract(args):
    enabled = getattr(args, 'center_evidence_enabled', False)
    freeze_student = getattr(args, 'center_evidence_freeze_student', False)
    if not isinstance(enabled, bool) or not isinstance(freeze_student, bool):
        raise ValueError('Center enable/freeze configuration must be boolean')
    mode = getattr(args, 'center_evidence_mode', 'fixed')
    if mode not in ('fixed', 'text_gate'):
        raise ValueError('Center evidence mode must be fixed or text_gate')
    if freeze_student and not enabled:
        raise ValueError('A frozen student requires an enabled center head')
    return {'enabled': enabled, 'mode': mode, 'freeze_student': freeze_student,
            'image_size': int(getattr(args, 'imsize', 224)),
            'hidden': int(getattr(args, 'center_evidence_hidden', 64)),
            'window_scale': float(getattr(args, 'center_evidence_window_scale', 3.)),
            'min_window_pixels': float(getattr(args, 'center_evidence_min_window_pixels', 64.)),
            'min_shift_pixels': float(getattr(args, 'center_evidence_min_shift_pixels', 12.)),
            'max_shift_pixels': float(getattr(args, 'center_evidence_max_shift_pixels', 32.))}


def parameter_group(name, args):
    """Return the optimizer group for a wrapper-qualified parameter name."""
    contract = center_contract(args)
    if name.startswith('center_head.') and contract['enabled']:
        if contract['mode'] == 'fixed' and name.startswith('center_head.route.'):
            return None
        return 'center'
    if name.startswith('student.') and not contract['freeze_student']:
        student_name = name[len('student.'):]
        if gentle_selected_name(student_name, args):
            return gentle_parameter_group(student_name)
    return None


def selected_name(name, args):
    return parameter_group(name, args) is not None


def assert_scope(model, args):
    if not isinstance(model, CenterEvidenceGrounder) or model._center_contract != center_contract(args):
        raise ValueError('Center wrapper configuration no longer matches its construction')
    assert_model_contract(model.student, args)
    wrong = [name for name, parameter in model.named_parameters()
             if parameter.requires_grad != selected_name(name, args)]
    if wrong:
        raise AssertionError(f'Unexpected center training scope: {wrong[:12]}')
    expected_clip = {name: parameter.requires_grad for name, parameter in model.student.base.clip.named_parameters()}
    if model.student.base._clip_control_trainability != expected_clip:
        raise AssertionError('PEFT adapter restore mask differs from the training scope')


class CenterEvidenceGrounder(nn.Module):
    """Add optional, trainable center evidence without changing the student API."""
    def __init__(self, student):
        super().__init__()
        if not isinstance(student, GentleDifferentialGrounder):
            raise TypeError('Expected a built GentleDifferentialGrounder student')
        assert_model_contract(student, student.args)
        self.student, self.args = student, student.args
        self._center_contract = center_contract(self.args)
        self.enabled = self._center_contract['enabled']
        self.freeze_student = self._center_contract['freeze_student']
        self.mode = self._center_contract['mode']
        self._check_adapter_restore()
        self.center_head = None
        if self.enabled:
            # Keep sampler / dropout RNG aligned with the disabled control.
            with torch.random.fork_rng(devices=[]):
                self.center_head = CenterEvidenceHead(
                    mode=self.mode, image_size=self._center_contract['image_size'],
                    hidden=self._center_contract['hidden'], window_scale=self._center_contract['window_scale'],
                    min_window_pixels=self._center_contract['min_window_pixels'],
                    min_shift_pixels=self._center_contract['min_shift_pixels'],
                    max_shift_pixels=self._center_contract['max_shift_pixels'])
        was_training = student.training
        self.configure_tuning()
        self.train(was_training)

    def _check_contract(self):
        if self._center_contract != center_contract(self.args):
            raise ValueError('Center configuration changed after model construction')
        assert_model_contract(self.student, self.args)

    def _check_adapter_restore(self):
        if not self.enabled or not getattr(self.student.base, 'open_lora', False):
            return
        if not getattr(self.args, 'match_infmae_lora_schedule', False):
            raise ValueError('Center training requires the PEFT adapter restore mask')
        blockers = ('infmae_bbox_head_only', 'infmae_frontend_only', 'clip_structure_new_only',
                    'exchange_train_new_only', 'exchange_projection_tuning')
        if (getattr(self.student.base, 'infmae_local_center', None) is not None
                or any(getattr(self.args, key, False) for key in blockers)):
            raise ValueError('Another tuning mode would override the PEFT adapter restore mask')

    def configure_tuning(self):
        self._check_contract()
        self._check_adapter_restore()
        self.student.configure_tuning()
        if self.freeze_student:
            for parameter in self.student.parameters():
                parameter.requires_grad_(False)
        self.student.base._clip_control_trainability = {
            name: parameter.requires_grad for name, parameter in self.student.base.clip.named_parameters()}
        if self.center_head is not None:
            for name, parameter in self.center_head.named_parameters():
                parameter.requires_grad_(not (self.mode == 'fixed' and name.startswith('route.')))
        assert_scope(self, self.args)

    def train(self, mode=True):
        self._check_contract()
        super().train(mode)
        # Frozen student features remain deterministic; in joint tuning the
        # gentle wrapper retains its last-four visual LoRA dropout policy.
        if mode and self.freeze_student:
            self.student.eval()
        return self

    def forward(self, img_data, text_data):
        self._check_contract()
        images, padding = img_data.tensors, img_data.mask
        # MMVG splits channels by mutating its NestedTensor; isolate the
        # student wrapper and retain the original normalized four channels.
        student_input = NestedTensor(images.clone(), padding.clone())
        if not self.enabled:
            return self.student(student_input, text_data)
        captured_text, captured_reg, handles = [], [], []

        def text_hook(module, inputs, output):
            if torch.is_tensor(output) and output.ndim == 2:
                captured_text.append(output)

        def vl_hook(module, inputs, output):
            if not torch.is_tensor(output) or output.ndim != 3 or output.shape[0] < 1:
                raise ValueError('Expected the VL encoder output as sequence,B,512')
            captured_reg.append(output[0])

        try:
            handles.append(self.student.base.clip.text_projection.register_forward_hook(text_hook))
            handles.append(self.student.base.vl_transformer.register_forward_hook(vl_hook))
            result = self.student(student_input, text_data)
        finally:
            for handle in handles:
                handle.remove()
        if not isinstance(result, (tuple, list)) or len(result) not in (5, 6):
            raise ValueError('Student must return its five-item output and optional auxiliary dictionary')
        if len(captured_text) != 1 or len(captured_reg) != 1:
            raise RuntimeError('Expected exactly one text EOS and one VL REG feature per forward')
        if captured_text[0].shape != (images.shape[0], 512) or captured_reg[0].shape != (images.shape[0], 512):
            raise ValueError('Captured text EOS and VL REG features must each have shape B,512')
        output, center_aux = self.center_head(images, padding, captured_text[0], captured_reg[0], result[0])
        main = (output, *result[1:5])
        if not self.training:
            return main
        if len(result) == 6 and not isinstance(result[5], dict):
            raise ValueError('Student auxiliary output must be a dictionary')
        auxiliary = dict(result[5]) if len(result) == 6 else {}
        if 'center_evidence' in auxiliary:
            raise ValueError('Student already contains center_evidence auxiliary output')
        auxiliary['center_evidence'] = center_aux
        return (*main, auxiliary)
