import json

from script_train.clip_cross_dataset.control import freeze_test_selections


def test_both_dataset_selections_are_sealed_before_evaluation(tmp_path):
    names=['cross_m3fd_seed13','cross_mfad_seed13']
    runs={name:{'epoch':i,'weight_view':'ema'} for i,name in enumerate(names)}
    for name in names:
        folder=tmp_path/'explore'/name
        folder.mkdir(parents=True)
        (folder/'best_checkpoint.pth').write_bytes(name.encode())
    selections=freeze_test_selections(tmp_path,names,runs,None)
    record=json.loads((tmp_path/'test_registration.json').read_text())
    assert set(record['selected'])==set(names)
    assert record['selection_used_test'] is False
    for name in names:
        assert (tmp_path/'frozen'/f'{name}.pth').read_bytes()==name.encode()
        assert selections[name]['epoch']==runs[name]['epoch']
    assert freeze_test_selections(tmp_path,names,runs,None)==selections
