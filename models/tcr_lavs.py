"""Target-conditioned source-key bias for the external LAVS exchange.

The frozen/finetuned C1 projections live in the original student.  A small
separate head reads only the existing visual/text features and never sees GT.
The optional target loss receives GT boxes in a separate training-only call.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .gentle_differential import (GentleDifferentialGrounder, assert_model_contract,
                                  parameter_group as gentle_group,
                                  selected_name as gentle_selected)


MODES = ('off', 'loss_only', 'bias')


def parameter_group(name, args):
    mode = getattr(args, 'tcr_mode', None)
    if mode not in MODES:
        raise ValueError('Unknown TCR mode')
    if name.startswith('tcr_target.') and mode != 'off':
        if name == 'tcr_target.bias_strength' and mode != 'bias':
            return None
        return 'tcr'
    if name.startswith('student.'):
        inner = name[len('student.'):]
        return gentle_group(inner) if gentle_selected(inner,args) else None
    return None


def selected_name(name,args):
    return parameter_group(name,args) is not None


def assert_scope(model,args):
    if not isinstance(model,TCRGrounder) or model.mode != getattr(args,'tcr_mode',None):
        raise ValueError('TCR model and recipe disagree')
    assert_model_contract(model.student,model.student.args)
    if (model.tcr_target is None) != (model.mode == 'off'):
        raise ValueError('TCR target branch and mode disagree')
    wrong=[n for n,p in model.named_parameters() if p.requires_grad != selected_name(n,args)]
    if wrong:
        raise AssertionError(f'Unexpected TCR tuning mask: {wrong[:12]}')
    expected={n:p.requires_grad for n,p in model.student.base.clip.named_parameters()}
    if model.student.base._clip_control_trainability != expected:
        raise AssertionError('PEFT adapter restore mask differs')


class TargetHead(nn.Module):
    """Cross-attend each patch to valid words, then estimate target relevance."""
    def __init__(self, channels, text_channels, hidden):
        super().__init__()
        self.visual_norm=nn.LayerNorm(channels)
        self.word_norm=nn.LayerNorm(text_channels)
        self.query=nn.Linear(channels,hidden)
        self.key=nn.Linear(text_channels,hidden)
        self.value=nn.Linear(text_channels,hidden)
        self.predict=nn.Sequential(nn.Linear(3*hidden,hidden),nn.GELU(),nn.Linear(hidden,1))
        # All patch priors begin equal, preserving the migrated LAVS function
        # exactly while allowing target supervision to shape the last layer.
        nn.init.zeros_(self.predict[-1].weight)
        nn.init.zeros_(self.predict[-1].bias)
        self.hidden=hidden

    def forward(self, tokens, words, word_padding, patch_padding):
        if tokens.ndim != 3 or words.ndim != 3 or tokens.shape[0] != words.shape[0]:
            raise ValueError('Expected batched visual and language tokens')
        patch=tokens[:,1:]
        batch,count,_=patch.shape
        if patch_padding.shape != (batch,count):
            raise ValueError('Visual padding must describe patch tokens only')
        if word_padding is None:
            word_padding=torch.zeros(words.shape[:2],device=words.device,dtype=torch.bool)
        if word_padding.shape != words.shape[:2]:
            raise ValueError('Text padding must have shape B,L')
        q=self.query(self.visual_norm(patch))
        normalized=self.word_norm(words)
        k,v=self.key(normalized),self.value(normalized)
        valid=~word_padding.bool()
        logits=(q @ k.transpose(-2,-1)) * (self.hidden**-.5)
        weights=logits.masked_fill(~valid[:,None,:],-torch.finfo(logits.dtype).max).softmax(-1)
        weights=weights * valid[:,None,:].to(weights.dtype)
        weights=weights/weights.sum(-1,keepdim=True).clamp_min(1e-12)
        context=weights @ v
        result=self.predict(torch.cat((q,context,q*context),dim=-1)).squeeze(-1)
        return result.masked_fill(patch_padding.bool(),0.)


def neutral_key_prior(logits,patch_padding):
    """Centered log prior: CLS and padding equal the valid patch mean.

    A spatially uniform target score gives an exactly zero bias for *all*
    keys, including CLS and preserved original foreground padding positions.
    """
    if logits.shape != patch_padding.shape:
        raise ValueError('Target logits and patch padding differ')
    valid=~patch_padding.bool()
    score=F.logsigmoid(logits.float())
    # Center differences from a valid reference, instead of summing identical
    # floating-point scores and subtracting their rounded mean: uniform inputs
    # then yield exactly zero, including when valid-patch counts are uneven.
    first=valid.to(torch.int64).argmax(dim=1,keepdim=True)
    relative=score-score.gather(1,first)
    mean=(relative*valid).sum(1,keepdim=True)/valid.sum(1,keepdim=True).clamp_min(1)
    score=(relative-mean).masked_fill(~valid,0.)
    return torch.cat((score.new_zeros(score.shape[0],1),score),dim=1)


def biased_delta(block,query,source,key_bias):
    """Apply key bias only to the migrated foreground read; return a residual.

    The untouched original read, including any reference read and output bias,
    is supplied by the student. For bias=0 this gives an exact zero residual.
    """
    q=block.split(block.query(query)*block.width**-.5,block.width)
    k=block.split(block.key(source),block.width)
    v=block.split(block.value(source),block.width)
    logits=q @ k.transpose(-2,-1)
    plain=logits.softmax(-1)
    revised=(logits+key_bias[:,None,None,:].to(logits.dtype)).softmax(-1)
    delta=(revised-plain) @ v
    merged=delta.transpose(1,2).reshape(query.shape[0],query.shape[1],block.channels)
    return F.linear(merged,block.output.weight,None)


class TargetKeyBias(nn.Module):
    def __init__(self,channels=768,text_channels=512,hidden=128):
        super().__init__()
        self.rgb=TargetHead(channels,text_channels,hidden)
        self.tir=TargetHead(channels,text_channels,hidden)
        # A projected optimizer step must clamp this parameter to [0,2];
        # clamp in forward additionally guarantees a nonnegative finite bias.
        # An initial positive coefficient avoids the zero boundary sticking
        # after projection; uniform zero-initialized maps still imply no bias.
        self.bias_strength=nn.Parameter(torch.tensor(.1))

    @torch.no_grad()
    def project_bias_strength(self):
        self.bias_strength.clamp_(0.,2.)

    def forward(self, exchange, inputs, output, *, apply_bias):
        rgb,tir,words,word_padding,padding=inputs
        if len(rgb)!=len(tir) or len(rgb)!=len(exchange.rgb) or len(tir)!=len(exchange.ir):
            raise ValueError('TCR requires paired registered visual levels')
        padding=padding.bool()
        rgb_logits=[self.rgb(r,words,word_padding,padding) for r in rgb]
        tir_logits=[self.tir(t,words,word_padding,padding) for t in tir]
        result_rgb,result_tir=output
        if apply_bias:
            strength=self.bias_strength.clamp(0.,2.)
            result_rgb=[old+biased_delta(block,r,t,strength*neutral_key_prior(logit,padding))
                        for old,block,r,t,logit in zip(result_rgb,exchange.rgb,rgb,tir,tir_logits)]
            result_tir=[old+biased_delta(block,t,r,strength*neutral_key_prior(logit,padding))
                        for old,block,r,t,logit in zip(result_tir,exchange.ir,rgb,tir,rgb_logits)]
        return (result_rgb,result_tir), {'rgb_logits':rgb_logits,'tir_logits':tir_logits,
                                        'patch_padding':padding}


class TCRGrounder(nn.Module):
    """Wrap a gentle-corrected student without changing its checkpoint keys."""
    def __init__(self,student,mode='off'):
        super().__init__()
        if not isinstance(student,GentleDifferentialGrounder):
            raise TypeError('TCR expects a GentleDifferentialGrounder student')
        assert_model_contract(student,student.args)
        if mode not in MODES or mode != getattr(student.args,'tcr_mode',None):
            raise ValueError('TCR mode must agree with student arguments')
        if student.args.experiment_case!='gentle_corrected':
            raise ValueError('TCR requires the corrected non-differential student')
        self.student,self.args,self.mode=student,student.args,mode
        exchange=student.base.differential_exchange
        self.tcr_target=None
        if mode!='off':
            block=exchange.rgb[0]
            # Do not change sampler or LoRA dropout RNG relative to no-head.
            with torch.random.fork_rng(devices=[]):
                self.tcr_target=TargetKeyBias(block.channels,block.reference_text.in_features,
                                              int(getattr(self.args,'tcr_hidden',128)))
        was_training=student.training
        self.configure_tuning()
        self.train(was_training)

    def configure_tuning(self):
        self.student.configure_tuning()
        if self.tcr_target is not None:
            for name,p in self.tcr_target.named_parameters():
                p.requires_grad_(not (name=='bias_strength' and self.mode!='bias'))
        assert_scope(self,self.student.args)

    def train(self,mode=True):
        super().train(mode)
        return self

    def forward(self,img_data,text_data):
        if self.mode != getattr(self.args,'tcr_mode',None):
            raise ValueError('TCR configuration changed after construction')
        if self.mode=='off' or (self.mode=='loss_only' and not self.training):
            return self.student(img_data,text_data)
        captured=[]
        def apply(module,inputs,output):
            result,aux=self.tcr_target(module,inputs,output,apply_bias=self.mode=='bias')
            captured.append(aux)
            return result
        handle=self.student.base.differential_exchange.register_forward_hook(apply)
        try:
            result=self.student(img_data,text_data)
        finally:
            handle.remove()
        if len(captured)!=1:
            raise RuntimeError('Expected exactly one external exchange forward')
        if not isinstance(result,(tuple,list)) or len(result) not in (5,6):
            raise ValueError('Unexpected student output')
        if not self.training:
            return tuple(result[:5])
        auxiliary=dict(result[5]) if len(result)==6 else {}
        if 'tcr_target' in auxiliary:
            raise ValueError('Student already contains TCR auxiliary data')
        auxiliary['tcr_target']=captured[0]
        return (*result[:5],auxiliary)


def target_alignment_loss(aux,target_boxes):
    """Box-derived soft targets for the *training* patch relevance heads.

    Balanced per-image positive/negative soft BCE prevents a large padding or
    background area from overwhelming small objects. No GT enters forward.
    """
    padding=aux['patch_padding'].bool()
    batch,count=padding.shape
    if target_boxes.shape!=(batch,4):
        raise ValueError('Target boxes must be B,4 normalized xywh')
    grid_size=math.isqrt(count)
    if grid_size*grid_size!=count:
        raise ValueError('TCR currently requires a square visual patch grid')
    target=target_boxes.float()
    finite=torch.isfinite(target).all(-1)
    eligible=finite & (target[:,:2]>=0).all(-1) & (target[:,:2]<1).all(-1) & (target[:,2:]>0).all(-1)
    safe=torch.where(finite[:,None],target,torch.zeros_like(target))
    y,x=torch.meshgrid(torch.arange(grid_size,device=target.device,dtype=target.dtype),
                       torch.arange(grid_size,device=target.device,dtype=target.dtype),indexing='ij')
    grid=torch.stack(((x+.5)/grid_size,(y+.5)/grid_size),-1).reshape(1,count,2)
    sigma=(safe[:,2:]*.25).clamp_min(.5/grid_size)
    soft=(-.5*((grid-safe[:,None,:2])/sigma[:,None,:]).square().sum(-1)).exp()
    valid=(~padding).to(soft.dtype)*eligible[:,None]
    pos=soft*valid
    neg=(1-soft)*valid
    eligible=eligible & (valid.sum(-1)>0) & (pos.sum(-1)>1e-12)
    total=target.new_zeros(())
    logits_list=aux['rgb_logits']+aux['tir_logits']
    if not logits_list:
        raise ValueError('Missing patch logits')
    for logits in logits_list:
        if logits.shape!=(batch,count):
            raise ValueError('Patch logits do not match the image grid')
        positive=-(F.logsigmoid(logits.float())*pos).sum(-1)/pos.sum(-1).clamp_min(1e-12)
        negative=-(F.logsigmoid(-logits.float())*neg).sum(-1)/neg.sum(-1).clamp_min(1e-12)
        total=total+((positive+negative)*.5*eligible).sum()/eligible.sum().clamp_min(1)
    return total/len(logits_list)
