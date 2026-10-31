"""Adapt the existing C1 visual adapters and fusion, with no extra inference path."""
import re

from torch import nn
from utils.misc import NestedTensor


def parameter_group(name):
    if name.startswith(('base.bbox_embed.', 'base.vl_transformer.encoder.norm.', 'base.reg_token.')):
        return 'decoder'
    if any(name.startswith(f'base.vl_transformer.encoder.layers.{i}.') for i in (3, 4, 5)):
        return 'decoder'
    if name.startswith(('base.cross_fusion_layers_vt.', 'base.cross_fusion_layers_tv.',
                        'base.visu_proj.', 'base.text_proj.', 'base.ml_visual_projection.',
                        'base.visu_token_norm.', 'base.visu_token_mlp.', 'base.iafv3_')):
        return 'fusion'
    if name.startswith('base.clip.') and '.vision_model.encoder.layers.' in name:
        match = re.search(r'encoder\.layers\.(\d+)\.', name)
        if (match and int(match.group(1)) in (8, 9, 10, 11)
                and ('.lora_A.' in name or '.lora_B.' in name)
                and ('.lora_rgb.' in name or '.lora_ir.' in name)):
            return 'visual_lora'
    return None


class LateAdaptationGrounder(nn.Module):
    def __init__(self, base, args):
        super().__init__()
        self.base, self.args = base, args
        self.configure_tuning()

    def configure_tuning(self):
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(parameter_group(name) is not None)
        # set_adapter otherwise freezes the inactive modality before backward.
        self.base._clip_control_trainability = {
            name: parameter.requires_grad for name, parameter in self.base.clip.named_parameters()}

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        # Keep fixed pretrained features deterministic. Only the adapters
        # being updated retain their original dropout regularization.
        if mode:
            for name, module in self.base.clip.named_modules():
                match = re.search(r'vision_model\.encoder\.layers\.(\d+)\.', name)
                if (match and int(match.group(1)) in (8, 9, 10, 11)
                        and '.lora_dropout.' in name and isinstance(module, nn.Dropout)):
                    module.train(True)
        return self

    def forward(self, img_data, text_data, target_boxes=None):
        if target_boxes is not None:
            raise ValueError('GT boxes must not enter the adaptation forward path')
        return self.base(NestedTensor(img_data.tensors, img_data.mask), text_data)
