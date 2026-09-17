import torch
import torch.nn as nn
import torch.nn.functional as F

from .language_model.bert import build_bert
from .visual_model.detr import build_detr
from .vl_transformer_attbalance import build_vl_transformer_attbalance


class AttBalance(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        hidden_dim = args.vl_hidden_dim
        divisor = 16 if args.dilation else 32
        if args.modality == "rgbt":
            self.num_visu_token = int((args.imsize / divisor) ** 2) * 2
        else:
            self.num_visu_token = int((args.imsize / divisor) ** 2)
        self.num_text_token = args.max_query_len

        self.visumodel = build_detr(args)
        self.textmodel = build_bert(args)

        num_total = self.num_visu_token + self.num_text_token + 1
        self.vl_pos_embed = nn.Embedding(num_total, hidden_dim)
        self.reg_token = nn.Embedding(1, hidden_dim)

        self.visu_proj = nn.Linear(self.visumodel.num_channels, hidden_dim)
        self.text_proj = nn.Linear(self.textmodel.num_channels, hidden_dim)

        self.vl_transformer = build_vl_transformer_attbalance(args)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        # Momentum branch.
        self.momentum = 0.9
        self.visumodel_m = build_detr(args)
        self.textmodel_m = build_bert(args)
        self.vl_pos_embed_m = nn.Embedding(num_total, hidden_dim)
        self.reg_token_m = nn.Embedding(1, hidden_dim)
        self.visu_proj_m = nn.Linear(self.visumodel.num_channels, hidden_dim)
        self.text_proj_m = nn.Linear(self.textmodel.num_channels, hidden_dim)
        self.vl_transformer_m = build_vl_transformer_attbalance(args)

        self.model_pairs = [
            [self.visumodel, self.visumodel_m],
            [self.textmodel, self.textmodel_m],
            [self.vl_pos_embed, self.vl_pos_embed_m],
            [self.reg_token, self.reg_token_m],
            [self.visu_proj, self.visu_proj_m],
            [self.text_proj, self.text_proj_m],
            [self.vl_transformer, self.vl_transformer_m],
        ]
        self.copy_params()

    def _forward_visual(self, img_data, visumodel):
        if self.args.modality == "rgbt":
            rgb_data = img_data.tensors[:, 0:3, :, :]
            thermal_img = img_data.tensors[:, 3:4, :, :].repeat(1, 3, 1, 1)
            ir_mask, ir_src = visumodel(thermal_img)
            rgb_mask, rgb_src = visumodel(rgb_data)
            visu_mask = torch.cat([rgb_mask, ir_mask], dim=1)
            visu_src = torch.cat([rgb_src, ir_src], dim=0)
        elif self.args.modality == "rgb":
            if img_data.tensors.shape[1] >= 4:
                rgb_data = img_data.tensors[:, 0:3, :, :]
                visu_mask, visu_src = visumodel(rgb_data)
            else:
                visu_mask, visu_src = visumodel(img_data)
        elif self.args.modality == "ir":
            c = img_data.tensors.shape[1]
            if c >= 4:
                ir_data = img_data.tensors[:, 3:4, :, :].repeat(1, 3, 1, 1)
                visu_mask, visu_src = visumodel(ir_data)
            elif c == 1:
                ir_data = img_data.tensors.repeat(1, 3, 1, 1)
                visu_mask, visu_src = visumodel(ir_data)
            elif c >= 3:
                # Keep modality semantics robust even if loader already outputs 3-channel IR.
                ir_data = img_data.tensors[:, 0:1, :, :].repeat(1, 3, 1, 1)
                visu_mask, visu_src = visumodel(ir_data)
            else:
                raise ValueError(f"Unsupported IR input channels: {c}")
        else:
            visu_mask, visu_src = visumodel(img_data)
        return visu_mask, visu_src

    def _forward_text(self, text_data, textmodel, text_proj):
        text_fea = textmodel(text_data)
        text_src, text_mask = text_fea.decompose()
        assert text_mask is not None
        text_src = text_proj(text_src)
        text_src = text_src.permute(1, 0, 2)
        text_mask = text_mask.flatten(1)
        return text_src, text_mask

    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)
                param_m.requires_grad = False

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1.0 - self.momentum)

    def forward(self, img_data, text_data):
        bs = img_data.tensors.shape[0]

        visu_mask, visu_src = self._forward_visual(img_data, self.visumodel)
        visu_src = self.visu_proj(visu_src)
        text_src, text_mask = self._forward_text(text_data, self.textmodel, self.text_proj)

        tgt_src = self.reg_token.weight.unsqueeze(1).repeat(1, bs, 1)
        tgt_mask = torch.zeros((bs, 1), dtype=torch.bool, device=tgt_src.device)
        vl_src = torch.cat([tgt_src, text_src, visu_src], dim=0)
        vl_mask = torch.cat([tgt_mask, text_mask, visu_mask], dim=1)
        vl_pos = self.vl_pos_embed.weight.unsqueeze(1).repeat(1, bs, 1)

        if self.training:
            with torch.no_grad():
                self._momentum_update()
                visu_mask_m, visu_src_m = self._forward_visual(img_data, self.visumodel_m)
                visu_src_m = self.visu_proj_m(visu_src_m)
                text_src_m, text_mask_m = self._forward_text(text_data, self.textmodel_m, self.text_proj_m)
                tgt_src_m = self.reg_token_m.weight.unsqueeze(1).repeat(1, bs, 1)
                tgt_mask_m = torch.zeros((bs, 1), dtype=torch.bool, device=tgt_src_m.device)
                vl_src_m = torch.cat([tgt_src_m, text_src_m, visu_src_m], dim=0)
                vl_mask_m = torch.cat([tgt_mask_m, text_mask_m, visu_mask_m], dim=1)
                vl_pos_m = self.vl_pos_embed_m.weight.unsqueeze(1).repeat(1, bs, 1)
                _, attn_m = self.vl_transformer_m(vl_src_m, vl_mask_m, vl_pos_m)
        else:
            attn_m = None

        vg_hs, attn = self.vl_transformer(vl_src, vl_mask, vl_pos)
        vg_hs = vg_hs[0]
        pred_box = self.bbox_embed(vg_hs).sigmoid()
        return pred_box, [attn, attn_m]


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
