"""Late-adaptation tuning scope for the migrated differential LAVS model.

Both variants use the correctly batched replacement architecture.  The only
architectural difference is whether its reference attention is enabled and
trainable.  In particular, the large migrated language foreground stays
frozen, matching the original late-adaptation scope.
"""
import re

from torch import nn

from .differential_lavs import DifferentialGrounder, DifferentialRead, contrast_key
from .late_adaptation import parameter_group as late_parameter_group


VARIANTS = ('gentle_corrected', 'gentle_differential')


def recipe_contract(args):
    """Reject a case label that disagrees with the actual model configuration."""
    case = getattr(args, 'experiment_case', None)
    if case not in VARIANTS:
        raise ValueError(f'Unregistered gentle differential variant: {case!r}')
    enabled = case == 'gentle_differential'
    if (getattr(args, 'replace_lavs', None) is not True
            or getattr(args, 'visual_exchange', None) != 'differential_lavs'
            or getattr(args, 'differential_enabled', None) is not enabled):
        raise ValueError(f'Model configuration disagrees with experiment_case {case!r}')
    seed = getattr(args, 'seed', None)
    if seed is None:
        raise ValueError('Experiment seed is required')
    return {'experiment_case': case, 'replace_lavs': True,
            'visual_exchange': 'differential_lavs', 'differential_enabled': enabled,
            'seed': int(seed)}


def parameter_group(name):
    """Match late_adaptation, with external LAVS mapped to its replacement."""
    if not name.startswith('base.'):
        return None
    key = name[5:]
    is_external = key.startswith('differential_exchange.')
    is_internal = '.language_replacement.' in key
    if is_external or is_internal:
        if contrast_key(name):
            return 'differential'
        # Encoder-internal language foreground was frozen in late_adaptation.
        return 'fusion' if is_external else None
    return late_parameter_group(name)


def selected_name(name, args):
    contract = recipe_contract(args)
    group = parameter_group(name)
    return group is not None and (group != 'differential' or contract['differential_enabled'])


def assert_model_contract(model, args):
    contract = recipe_contract(args)
    if getattr(model, '_gentle_contract', None) != contract:
        raise ValueError('Model was built for a different gentle experiment or seed')
    if recipe_contract(model.args) != contract:
        raise ValueError('Model arguments no longer match the registered experiment')
    base_args = getattr(model.base, 'args', args)
    if recipe_contract(base_args) != contract:
        raise ValueError('Underlying model arguments disagree with the wrapper')
    reads = [module for module in model.base.modules() if isinstance(module, DifferentialRead)]
    if not reads or any(module.enabled != contract['differential_enabled'] for module in reads):
        raise ValueError('Built attention modules disagree with experiment_case')
    return contract


class GentleDifferentialGrounder(DifferentialGrounder):
    def __init__(self, base, args):
        # The immutable copy catches later mutations of the shared args object.
        object.__setattr__(self, '_gentle_contract', recipe_contract(args))
        super().__init__(base, args)

    def configure_tuning(self):
        assert_model_contract(self, self.args)
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(selected_name(name, self.args))
        # PEFT's adapter switch otherwise changes requires_grad on the other
        # modality. Preserve the exact registered mask for both adapters.
        self.base._clip_control_trainability = {
            name: parameter.requires_grad for name, parameter in self.base.clip.named_parameters()}

    def train(self, mode=True):
        assert_model_contract(self, self.args)
        # Match LateAdaptationGrounder: deterministic frozen modules while the
        # late vision LoRA dropout remains active during training.
        nn.Module.train(self, mode)
        self.base.eval()
        if mode:
            for name, module in self.base.clip.named_modules():
                match = re.search(r'vision_model\.encoder\.layers\.(\d+)\.', name)
                if (match and int(match.group(1)) in (8, 9, 10, 11)
                        and '.lora_dropout.' in name and isinstance(module, nn.Dropout)):
                    module.train(True)
        return self
