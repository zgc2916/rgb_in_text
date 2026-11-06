import json
from pathlib import Path

import pytest

from script_train.clip_gentle_differential_control.control import freeze_test_selections


def test_diagnostic_seals_all_validation_selections_even_when_gate_fails(tmp_path):
    names=['gentle_corrected_seed13','gentle_differential_seed13']
    runs={names[0]:{'epoch':-1,'weight_view':'raw'}, names[1]:{'epoch':4,'weight_view':'ema'}}
    for name in names:
        directory=tmp_path/'explore'/name
        directory.mkdir(parents=True)
        (directory/'best_checkpoint.pth').write_bytes(name.encode())
    selections=freeze_test_selections(tmp_path,names,runs,selected=None)
    registration=json.loads((tmp_path/'test_registration.json').read_text())
    assert registration['selected_variant'] is None
    assert registration['selection_used_test'] is False
    assert registration['validation_gate_unchanged'] is True
    assert set(selections)==set(names)
    for name in names:
        assert Path(selections[name]['checkpoint']).read_bytes()==name.encode()
    assert freeze_test_selections(tmp_path,names,runs,selected=None)==selections


def test_resume_rejects_changed_checkpoint_and_metadata(tmp_path):
    name='gentle_differential_seed13'
    directory=tmp_path/'explore'/name
    directory.mkdir(parents=True)
    source=directory/'best_checkpoint.pth';source.write_bytes(b'original')
    runs={name:{'epoch':4,'weight_view':'ema'}}
    freeze_test_selections(tmp_path,[name],runs,selected=None)
    runs[name]['epoch']=5
    with pytest.raises(ValueError,match='Sealed test selection changed'):
        freeze_test_selections(tmp_path,[name],runs,selected=None)
    runs[name]['epoch']=4
    source.write_bytes(b'changed')
    with pytest.raises(ValueError,match='frozen checkpoint differs'):
        freeze_test_selections(tmp_path,[name],runs,selected=None)
