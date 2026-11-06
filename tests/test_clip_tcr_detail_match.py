import pytest
from script_train.clip_tcr_detail_match import runtime as r


@pytest.mark.parametrize('method,detail,boundary',[('detail',True,False),('boundary',False,True),('joint',True,True)])
def test_variant_keeps_matched_tcr_recipe(tmp_path,method,detail,boundary):
    from script_train.clip_full_match.runtime import recipe as reference
    a=r.recipe(tmp_path/'candidate','m3fd',method)
    b=reference(tmp_path/'candidate','m3fd','tcr_query')
    for name in ('dataset','epochs','seed','batch_size','grad_accum_steps','lr','weight_decay','optimizer',
                 'lr_scheduler','use_contrastive_loss','use_rtcc_constrain_loss','use_mask_loss',
                 'tcr_mode','tcr_query_enabled','tcr_loss_weight','retrain','match_infmae_lora_schedule'):
        assert getattr(a,name)==getattr(b,name)
    assert (a.enable_tir_detail,a.enable_boundary_refiner)==(detail,boundary)
    assert a.epochs==120 and a.full_detail_variant==method


def test_mismatched_flags_are_rejected_before_build(tmp_path):
    a=r.recipe(tmp_path,'m3fd','detail');a.enable_boundary_refiner=True
    with pytest.raises(ValueError,match='flags'):r.build(a)
