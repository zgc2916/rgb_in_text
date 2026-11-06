"""Word conditioning and cross-level deformable RGB/TIR exchange.

These modules replace both parts of paper LAVS. They do not call, reuse,
or retain the original language-visual or RGB/TIR attention projections.
"""
import math
import re

import torch
from torch import nn
from torch.nn import functional as F

from utils.misc import NestedTensor


def legacy_lavs_key(name):
    return (name.startswith(('cross_fusion_layers_vt.', 'cross_fusion_layers_tv.'))
            or (name.startswith('clip.') and any('.'+x+'.' in name for x in (
                'cross_attn_st','cross_attn_sv','cross_norm_st','cross_norm_sv','cross_mlp_st','cross_mlp_sv'))))


def replacement_key(name):
    return name.startswith('replacement_exchange.') or '.word_conditioner.' in name


def parameter_group(name):
    if not name.startswith('base.'):
        return None
    key=name[5:]
    if replacement_key(key):
        return 'replacement'
    if legacy_lavs_key(key):
        return 'original_lavs'
    if key.startswith(('vl_transformer.', 'bbox_embed.', 'reg_token.', 'visu_proj.', 'text_proj.',
                       'condition_text_proj.', 'ml_visual_projection.', 'visu_token_norm.',
                       'visu_token_mlp.', 'iafv3_')):
        return 'grounding'
    if key.startswith('clip.') and '.vision_model.' in key and ('.lora_A.' in key or '.lora_B.' in key):
        return 'visual_lora'
    return None


class WordConditioningBlock(nn.Module):
    """Visual queries read individual valid words, rather than A.T @ A @ V."""
    def __init__(self, channels=768, hidden=128):
        super().__init__()
        self.visual_norm=nn.LayerNorm(channels)
        self.word_norm=nn.LayerNorm(channels)
        # Names intentionally avoid PEFT's q_proj/k_proj/v_proj target pattern.
        self.query=nn.Linear(channels,hidden)
        self.key=nn.Linear(channels,hidden)
        self.value=nn.Linear(channels,hidden)
        self.output=nn.Linear(hidden,channels)
        self.modality_scale=nn.Parameter(torch.zeros(2))
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, visual, words, word_padding, modality):
        if word_padding is None or word_padding.shape!=words.shape[:2]:
            raise ValueError('Word padding must match B,T text features')
        normalized=self.word_norm(words)
        q=self.query(self.visual_norm(visual))[:,None]
        k,v=self.key(normalized)[:,None],self.value(normalized)[:,None]
        valid=~word_padding.bool()
        support=valid.any(-1)
        allowed=valid.clone()
        allowed[:,0] |= ~support
        context=F.scaled_dot_product_attention(q,k,v,attn_mask=allowed[:,None,None,:],dropout_p=0.)[:,0]
        update=self.output(context)*support[:,None,None]
        direction=0 if modality=='rgb' else 1
        return visual+self.modality_scale[direction].sigmoid()*update


def reference_grid(side, device, dtype):
    y,x=torch.meshgrid(torch.arange(side,device=device),torch.arange(side,device=device),indexing='ij')
    return torch.stack((2*(x+.5)/side-1,2*(y+.5)/side-1),dim=-1).reshape(-1,2).to(dtype)


def sample_heads(value, grid, side, heads):
    """B,N,D + B,Q,H,P,2 -> B,Q,H,P,D/H; pixel centers use align_corners=False."""
    batch,_,channels=value.shape
    width=channels//heads
    image=value.reshape(batch,side,side,heads,width).permute(0,3,4,1,2).reshape(batch*heads,width,side,side)
    points=grid.permute(0,2,1,3,4).reshape(batch*heads,grid.shape[1],grid.shape[3],2)
    sampled=F.grid_sample(image,points,mode='bilinear',padding_mode='zeros',align_corners=False)
    return sampled.reshape(batch,heads,width,grid.shape[1],grid.shape[3]).permute(0,3,1,4,2)


class CrossLevelDeformableExchange(nn.Module):
    """Language-conditioned, simultaneous cross-modal reads across four depths.

    CLIP levels share spatial resolution. 'Levels' here means semantic depths,
    not a fabricated image pyramid. Fixed and learned-offset controls share
    the same projections, words, sampling count and output interface.
    """
    def __init__(self, channels=768, text_channels=512, hidden=128, heads=4, levels=4,
                 learn_offsets=True, offset_bound=2.):
        super().__init__()
        if hidden%heads or levels<1:
            raise ValueError('Hidden dimension must be divisible by heads')
        self.hidden,self.heads,self.levels=hidden,heads,levels
        self.points,self.offset_bound,self.learn_offsets=5,offset_bound,learn_offsets
        self.norms=nn.ModuleList(nn.LayerNorm(channels) for _ in range(levels))
        self.projections=nn.ModuleList(nn.Linear(channels,hidden) for _ in range(levels))
        self.word_norm=nn.LayerNorm(text_channels)
        self.word_keys=nn.Linear(text_channels,hidden)
        self.word_values=nn.Linear(text_channels,hidden)
        self.queries=nn.ModuleList(nn.Linear(3*hidden,hidden) for _ in range(levels))
        slots=heads*levels*self.points
        self.offsets=nn.ModuleList(nn.Linear(hidden,2*slots) for _ in range(levels))
        self.scores=nn.ModuleList(nn.Linear(hidden,slots) for _ in range(levels))
        self.outputs=nn.ModuleList(nn.Linear(hidden,channels) for _ in range(levels))
        self.direction=nn.Parameter(torch.zeros(2,hidden))
        self.register_buffer('pattern',torch.tensor([[0.,0.],[-1.,0.],[1.,0.],[0.,-1.],[0.,1.]]),persistent=False)
        for collection in (self.offsets,self.scores,self.outputs):
            for layer in collection:
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, rgb, tir, words, word_padding, padding):
        if len(rgb)!=self.levels or len(tir)!=self.levels:
            raise ValueError('Unexpected number of semantic levels')
        if words is None or word_padding is None:
            raise ValueError('Full text tokens and their padding are required')
        batch,tokens,_=rgb[0].shape
        side=math.isqrt(tokens-1)
        if side**2!=tokens-1 or padding.shape!=(batch,tokens-1):
            raise ValueError('Expected matching square grids')
        valid=(~padding.bool()).to(rgb[0].dtype)
        normalized_words=self.word_norm(words)
        wk,wv=self.word_keys(normalized_words),self.word_values(normalized_words)
        word_valid=~word_padding.bool()
        word_support=word_valid.any(-1)
        word_valid=word_valid.clone()
        word_valid[:,0] |= ~word_support
        rp=[p(n(x)) for p,n,x in zip(self.projections,self.norms,rgb)]
        tp=[p(n(x)) for p,n,x in zip(self.projections,self.norms,tir)]
        base=reference_grid(side,rgb[0].device,rgb[0].dtype)
        width=self.hidden//self.heads

        def read(own,donor,original,direction):
            outputs=[]
            for level,x in enumerate(own):
                word_weights=(x@wk.transpose(1,2)/math.sqrt(self.hidden)).masked_fill(~word_valid[:,None,:],-1e4).softmax(-1)
                context=(word_weights@wv)*word_support[:,None,None]
                query=self.queries[level](torch.cat((x,donor[level],context),dim=-1))+self.direction[direction]
                patch_query=query[:,1:]
                shape=(batch,tokens-1,self.heads,self.levels,self.points)
                displacement=self.offsets[level](patch_query).reshape(*shape,2).tanh()*self.offset_bound
                if not self.learn_offsets:
                    displacement=displacement*0.
                grid=base[None,:,None,None,None,:]+(self.pattern[None,None,None,None,:,:]+displacement)*2/side
                sampled=[]
                support=[]
                # One donor feature map per encoder depth, same pixel geometry.
                for source in range(self.levels):
                    points=grid[:,:,:,source]
                    sampled.append(sample_heads(donor[source][:,1:]*valid[:,:,None],points,side,self.heads))
                    validity=valid[:,:,None].expand(-1,-1,self.heads)
                    support.append(sample_heads(validity,points,side,self.heads)[...,0].clamp(0,1))
                samples=torch.stack(sampled,dim=3)
                coverage=torch.stack(support,dim=3)
                q=patch_query.reshape(batch,tokens-1,self.heads,width)
                logits=(samples*q[:,:,:,None,None,:]).sum(-1)/math.sqrt(width)
                logits=logits+self.scores[level](patch_query).reshape(*shape)
                logits=logits+coverage.clamp_min(1e-6).log()
                weights=logits.flatten(-2).softmax(-1).reshape_as(logits)*coverage
                weights=weights/weights.sum((-1,-2),keepdim=True).clamp_min(1e-6)
                message=(samples*weights[...,None]).sum((3,4)).reshape(batch,tokens-1,self.hidden)
                update=self.outputs[level](message)*valid[:,:,None]
                # CLS reads global, text-selected donor evidence; no original CA.
                donor_patches=torch.cat([d[:,1:] for d in donor],dim=1)
                donor_valid=valid.repeat(1,self.levels)
                global_score=(donor_patches*query[:,:1]).sum(-1)/math.sqrt(self.hidden)
                global_weight=global_score.masked_fill(~donor_valid.bool(),-1e4).softmax(-1)*donor_valid
                global_weight=global_weight/global_weight.sum(-1,keepdim=True).clamp_min(1e-6)
                global_message=(donor_patches*global_weight[:,:,None]).sum(1,keepdim=True)
                cls_update=self.outputs[level](global_message)*valid.any(-1)[:,None,None]
                outputs.append(original[level]+torch.cat((cls_update,update),dim=1))
            return outputs

        return read(rp,tp,rgb,0),read(tp,rp,tir,1)


class LavsReplacementGrounder(nn.Module):
    def __init__(self,base,args):
        super().__init__()
        self.base,self.args=base,args
        self.configure_tuning()

    def configure_tuning(self):
        for name,p in self.named_parameters():
            selected=parameter_group(name) is not None
            if self.args.replace_lavs and self.args.replacement_fixed_offsets and '.offsets.' in name:
                selected=False
            p.requires_grad_(selected)
        self.base._clip_control_trainability={n:p.requires_grad for n,p in self.base.clip.named_parameters()}

    def train(self,mode=True):
        super().train(mode)
        self.base.eval()
        if mode:
            for name,module in self.base.clip.named_modules():
                if '.vision_model.' in name and '.lora_dropout.' in name and isinstance(module,nn.Dropout):
                    module.train(True)
        return self

    def forward(self,img_data,text_data,target_boxes=None):
        if target_boxes is not None:
            raise ValueError('GT boxes are forbidden in the inference path')
        return self.base(NestedTensor(img_data.tensors,img_data.mask),text_data)
