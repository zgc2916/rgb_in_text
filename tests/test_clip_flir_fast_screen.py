import pytest
from script_train.clip_flir_fast_screen import runtime as r


@pytest.mark.parametrize('method,detail,boundary',[('tcr',False,False),('detail',True,False),
                                                   ('boundary',False,True),('joint',True,True)])
def test_short_screen_cases_have_matched_training(tmp_path,method,detail,boundary):
    a=r.recipe(tmp_path, 'flir',method)
    b=r.recipe(tmp_path, 'flir','tcr')
    for name in ('dataset','epochs','seed','batch_size','grad_accum_steps','lr','weight_decay','optimizer',
                 'use_contrastive_loss','use_rtcc_constrain_loss','use_mask_loss','tcr_query_enabled',
                 'tcr_loss_weight','screen_source','screen_source_sha'):
        assert getattr(a,name)==getattr(b,name)
    assert (a.enable_tir_detail,a.enable_boundary_refiner)==(detail,boundary)
    assert a.dataset=='rgbtvg_flir' and a.epochs==10 and a.retrain==''


def test_inconsistent_combination_rejected_before_model_construction(tmp_path):
    a=r.recipe(tmp_path,'flir','detail');a.enable_boundary_refiner=True
    with pytest.raises(ValueError,match='flags'):r.build(a)
