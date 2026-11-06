import json
from pathlib import Path

import pytest

from script_train.clip_center_evidence_control.control import freeze_test_selections


def test_all_three_center_controls_are_frozen_before_diagnostic_testing(tmp_path):
    names=['center_none_seed13','center_fixed_seed13','center_textgate_seed13']
    runs={name:{'epoch':i-1,'weight_view':'raw' if i!=2 else 'ema'} for i,name in enumerate(names)}
    for name in names:
        directory=tmp_path/'explore'/name
        directory.mkdir(parents=True)
        (directory/'best_checkpoint.pth').write_bytes(name.encode())
    frozen=freeze_test_selections(tmp_path,names,runs,selected=None)
    registered=json.loads((tmp_path/'test_registration.json').read_text())
    assert set(registered['selected'])==set(names)
    assert registered['selection_used_test'] is False
    assert registered['validation_gate_unchanged'] is True
    for name in names:
        assert Path(frozen[name]['checkpoint']).read_bytes()==name.encode()
    Path(frozen[names[1]]['checkpoint']).write_bytes(b'bad')
    with pytest.raises(ValueError,match='frozen checkpoint differs'):
        freeze_test_selections(tmp_path,names,runs,selected=None)
