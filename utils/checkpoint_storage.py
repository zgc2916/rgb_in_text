"""Lossless storage of complete models relative to an immutable C1 checkpoint."""
from collections import OrderedDict
import copy
import hashlib
from pathlib import Path

import torch


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def pack_relative(checkpoint, base_model, base_path, base_sha256):
    if 'model' not in checkpoint or any(k in checkpoint for k in ('model_storage', 'model_delta')):
        raise ValueError('Expected a complete model without an existing delta format')
    model = checkpoint['model']
    changed = {name: value for name, value in model.items()
               if name not in base_model or value.shape != base_model[name].shape
               or value.dtype != base_model[name].dtype or not torch.equal(value, base_model[name])}
    result = {key: value for key, value in checkpoint.items() if key != 'model'}
    result.update(model_delta=changed, model_storage={
        'kind': 'exact_c1_relative_v1', 'base_checkpoint': str(Path(base_path).resolve()),
        'base_sha256': base_sha256, 'model_keys': list(model),
        'state_dict_metadata': copy.deepcopy(getattr(model, '_metadata', None))})
    return result


def unpack_relative(checkpoint, base_model):
    storage = checkpoint['model_storage']
    if storage['kind'] != 'exact_c1_relative_v1':
        raise ValueError('Unknown compressed model format')
    delta = checkpoint['model_delta']
    if set(delta) - set(storage['model_keys']):
        raise ValueError('Delta contains keys outside the original model schema')
    model = OrderedDict((name, delta[name] if name in delta else base_model[name])
                        for name in storage['model_keys'])
    if storage['state_dict_metadata'] is not None:
        model._metadata = copy.deepcopy(storage['state_dict_metadata'])
    result = {key: value for key, value in checkpoint.items() if key not in ('model_storage', 'model_delta')}
    result['model'] = model
    return result


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if 'model_storage' not in checkpoint:
        return checkpoint
    storage = checkpoint['model_storage']
    if file_sha256(storage['base_checkpoint']) != storage['base_sha256']:
        raise ValueError('The immutable C1 reconstruction checkpoint changed')
    base = torch.load(storage['base_checkpoint'], map_location='cpu', weights_only=False)
    return unpack_relative(checkpoint, base['model'])
