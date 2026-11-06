"""Query-conditioned soft target priors for external RGB/TIR attention.

The matched global variant uses the same target heads and same initial bias
strength. The query variant only scales each source-key prior by the receiving
patch's target relevance, preserving context for less relevant queries.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .tcr_lavs import TCRGrounder, TargetKeyBias, neutral_key_prior


def query_gate(logits, padding):
    """Bounded receiving-patch gate, with neutral CLS and padded tokens."""
    if logits.shape != padding.shape:
        raise ValueError('Target logits and visual padding must match')
    gate=2*torch.sigmoid(logits.float())
    gate=gate.masked_fill(padding.bool(),1.)
    return torch.cat((gate.new_ones(gate.shape[0],1),gate),dim=1)


def query_biased_delta(block,query,source,prior,gate):
    """Difference between query-conditioned and original foreground reads."""
    if prior.shape != source.shape[:2] or gate.shape != query.shape[:2]:
        raise ValueError('Query gate or source prior shape differs from tokens')
    q=block.split(block.query(query)*block.width**-.5,block.width)
    k=block.split(block.key(source),block.width)
    v=block.split(block.value(source),block.width)
    logits=q @ k.transpose(-2,-1)
    original=logits.softmax(-1)
    bias=(gate[:,:,None]*prior[:,None,:])[:,None].to(logits.dtype)
    modified=(logits+bias).softmax(-1)
    delta=(modified-original) @ v
    merged=delta.transpose(1,2).reshape(query.shape[0],query.shape[1],block.channels)
    return F.linear(merged,block.output.weight,None)


class QueryTargetKeyBias(TargetKeyBias):
    def __init__(self,channels=768,text_channels=512,hidden=128,query_enabled=False):
        super().__init__(channels,text_channels,hidden)
        self.query_enabled=bool(query_enabled)

    def forward(self,exchange,inputs,output,*,apply_bias):
        if not (apply_bias and self.query_enabled):
            return super().forward(exchange,inputs,output,apply_bias=apply_bias)
        rgb,tir,words,word_padding,padding=inputs
        if len(rgb)!=len(tir) or len(rgb)!=len(exchange.rgb) or len(tir)!=len(exchange.ir):
            raise ValueError('Expected paired RGB/TIR visual levels')
        padding=padding.bool()
        rgb_logits=[self.rgb(r,words,word_padding,padding) for r in rgb]
        tir_logits=[self.tir(t,words,word_padding,padding) for t in tir]
        strength=self.bias_strength.clamp(0.,2.)
        result_rgb,result_tir=output
        result_rgb=[old+query_biased_delta(block,r,t,
                    strength*neutral_key_prior(tl,padding),query_gate(rl,padding))
                    for old,block,r,t,rl,tl in zip(result_rgb,exchange.rgb,rgb,tir,rgb_logits,tir_logits)]
        result_tir=[old+query_biased_delta(block,t,r,
                    strength*neutral_key_prior(rl,padding),query_gate(tl,padding))
                    for old,block,r,t,rl,tl in zip(result_tir,exchange.ir,rgb,tir,rgb_logits,tir_logits)]
        return (result_rgb,result_tir),{'rgb_logits':rgb_logits,'tir_logits':tir_logits,
                                        'patch_padding':padding}


class QueryTCRGrounder(TCRGrounder):
    """Keep student and trainable head keys identical to TCR, add a gate switch."""
    def __init__(self,student,mode='off'):
        super().__init__(student,mode)
        query_enabled=bool(getattr(student.args,'tcr_query_enabled',False))
        initial=float(getattr(student.args,'tcr_initial_strength',.5))
        if not 0 < initial <= 2:
            raise ValueError('TCR initial strength must lie in (0, 2]')
        if query_enabled and mode!='bias':
            raise ValueError('A query gate requires the bias mode')
        if self.tcr_target is not None:
            old=self.tcr_target
            with torch.random.fork_rng(devices=[]):
                head=QueryTargetKeyBias(old.rgb.visual_norm.normalized_shape[0],
                    old.rgb.word_norm.normalized_shape[0],old.rgb.hidden,query_enabled)
            head.load_state_dict(old.state_dict(),strict=True)
            with torch.no_grad():
                head.bias_strength.fill_(initial)
            self.tcr_target=head
            self.configure_tuning()
            self.train(student.training)
        self.query_enabled=query_enabled

    def forward(self,*args,**kwargs):
        if self.query_enabled != bool(getattr(self.args,'tcr_query_enabled',False)):
            raise ValueError('Query gate setting changed after construction')
        return super().forward(*args,**kwargs)
