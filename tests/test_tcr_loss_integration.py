from types import ModuleType, SimpleNamespace
import sys

import pytest
import torch

from utils.loss_utils import trans_vg_loss


def boxes():
    return torch.tensor([[.5, .5, .2, .3]], requires_grad=True)


def args(**values):
    return SimpleNamespace(infmae_bbox_head_only=True, **values)


def run(config, prediction, target, aux=None):
    return trans_vg_loss(config, prediction, target, None, None, aux=aux)


def test_default_and_off_preserve_existing_loss():
    target=boxes().detach()
    a=run(args(),boxes(),target)
    b=run(args(tcr_mode='off'),boxes(),target)
    assert set(a)==set(b)=={'loss_bbox','loss_giou'}
    assert all(torch.equal(a[k],b[k]) for k in a)


@pytest.mark.parametrize('mode',['loss_only','bias'])
def test_registered_modes_forward_aux_targets_and_backpropagate(monkeypatch, mode):
    module=ModuleType('models.tcr_lavs')
    observed={}
    def target_loss(aux, targets):
        observed.update(aux=aux,targets=targets)
        return (aux-targets[:, :1]).square().mean()
    module.target_alignment_loss=target_loss
    monkeypatch.setitem(sys.modules,'models.tcr_lavs',module)
    target=boxes().detach()
    aux=torch.tensor([[.2]],requires_grad=True)
    result=run(args(tcr_mode=mode,tcr_loss_weight=.05),boxes(),target,{'tcr_target':aux})
    assert observed['aux'] is aux and observed['targets'] is target
    assert torch.allclose(result['loss_tcr_target'],.05*(aux-target[:, :1]).square().mean())
    result['loss_tcr_target'].backward()
    assert aux.grad is not None and aux.grad.abs().sum()>0


def test_missing_aux_invalid_mode_and_nonfinite_weight_rejected(monkeypatch):
    module=ModuleType('models.tcr_lavs')
    module.target_alignment_loss=lambda aux,target: target.sum()*0
    monkeypatch.setitem(sys.modules,'models.tcr_lavs',module)
    target=boxes().detach()
    with pytest.raises(ValueError,match='auxiliary'):
        run(args(tcr_mode='bias'),boxes(),target)
    with pytest.raises(ValueError,match='Unsupported'):
        run(args(tcr_mode='unknown'),boxes(),target)
    with pytest.raises(ValueError,match='tcr_loss_weight'):
        run(args(tcr_mode='bias',tcr_loss_weight=float('nan')),boxes(),target,{'tcr_target':{}})
