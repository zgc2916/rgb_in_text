import copy
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.nn as nn
from script_train.clip_full_match import runtime as r


def test_matched_recipes_keep_original_training_settings(tmp_path):
    a=r.recipe(tmp_path/'one','m3fd','c1');b=r.recipe(tmp_path/'two','m3fd','tcr_query')
    assert a.dataset==b.dataset=='rgbtvg_m3fd'
    for field in ('epochs','lr','weight_decay','batch_size','grad_accum_steps','use_contrastive_loss',
                  'use_rtcc_constrain_loss','use_mask_loss','match_infmae_lora_schedule','retrain','seed'):
        assert getattr(a,field)==getattr(b,field)
    assert a.epochs==120 and (a.batch_size,a.grad_accum_steps)==(12,4)
    assert all((a.use_contrastive_loss,a.use_rtcc_constrain_loss,a.use_mask_loss))
    assert a.replace_lavs is False and b.replace_lavs is True
    assert Path(a.retrain).is_file()
    assert 'output_clip_' not in a.retrain


class Exchange(nn.Module):
    def __init__(self):
        super().__init__()
        from models.differential_lavs import DifferentialRead
        self.rgb=nn.ModuleList([DifferentialRead(channels=12,heads=3,context_channels=8,enabled=False)])
        self.ir=nn.ModuleList([DifferentialRead(channels=12,heads=3,context_channels=8,enabled=False)])
    def forward(self,rgb,tir,words,word_padding,padding):
        return rgb,tir


class Tiny(nn.Module):
    def __init__(self):super().__init__();self.differential_exchange=Exchange()
    def forward(self,x):
        rgb,tir,words,padding=x
        out=self.differential_exchange(rgb,tir,words,torch.zeros(words.shape[:2],dtype=torch.bool),padding)
        return (out[0][0],None,None,None,None)


def test_full_match_tcr_adds_aux_only_during_training(monkeypatch):
    from models import tcr_query_lavs
    real=tcr_query_lavs.QueryTargetKeyBias
    monkeypatch.setattr(tcr_query_lavs,'QueryTargetKeyBias',lambda **kw: real(channels=12,text_channels=8,hidden=4,query_enabled=True))
    model=r.install_tcr(Tiny(),SimpleNamespace(tcr_query_enabled=True,tcr_hidden=4,tcr_initial_strength=.5))
    x=([torch.randn(2,5,12)],[torch.randn(2,5,12)],torch.randn(2,3,8),torch.zeros(2,4,dtype=torch.bool))
    assert len(model(x))==6 and 'tcr_target' in model(x)[5]
    model.eval();assert len(model(x))==5
