from collections import OrderedDict
from types import SimpleNamespace

import pytest
import torch

from utils.checkpoint_storage import pack_relative, unpack_relative, load_checkpoint, file_sha256


def test_relative_storage_preserves_changed_new_removed_and_different_dtype_tensors():
    base = {'same': torch.ones(2), 'changed': torch.zeros(2), 'removed': torch.zeros(1),
            'dtype': torch.ones(2, dtype=torch.float32)}
    model = OrderedDict(same=torch.ones(2), changed=torch.ones(2), new=torch.ones(1),
                        dtype=torch.ones(2, dtype=torch.float64))
    model._metadata = {'': {'version': 2}}
    checkpoint = {'model': model, 'epoch': 19, 'args': SimpleNamespace(seed=13)}
    compact = pack_relative(checkpoint, base, '/tmp/example.pth', 'test')
    assert set(compact['model_delta']) == {'changed', 'new', 'dtype'}
    restored = unpack_relative(compact, base)
    assert list(restored['model']) == list(model)
    assert restored['model']._metadata == model._metadata
    assert all(torch.equal(value, restored['model'][name]) and value.dtype == restored['model'][name].dtype
               for name, value in model.items())
    assert restored['args'].seed == 13 and restored['epoch'] == 19


def test_compressed_loading_rejects_a_modified_reconstruction_base(tmp_path):
    base = tmp_path / 'base.pth'
    torch.save({'model': {'weight': torch.zeros(2)}}, base)
    source = {'model': {'weight': torch.ones(2)}, 'epoch': 1}
    packed = pack_relative(source, {'weight': torch.zeros(2)}, base, file_sha256(base))
    path = tmp_path / 'compact.pth'
    torch.save(packed, path)
    assert torch.equal(load_checkpoint(path)['model']['weight'], torch.ones(2))
    torch.save({'model': {'weight': torch.ones(2)}}, base)
    with pytest.raises(ValueError, match='changed'):
        load_checkpoint(path)


def test_complete_legacy_checkpoint_keeps_the_existing_loading_behavior(tmp_path):
    path = tmp_path / 'legacy.pth'
    torch.save({'model': {'weight': torch.ones(2)}, 'epoch': 3}, path)
    assert load_checkpoint(path)['epoch'] == 3
