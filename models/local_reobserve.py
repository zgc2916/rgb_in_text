"""Two text-conditioned native-image reads with a preserved global prediction.

This is a bounded local enhancement of C1, not a complete LAVS replacement.
Crop geometry is computed from predictions and image geometry only. Ground
truth is consumed separately by the training loss, never by this forward path.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from utils.misc import NestedTensor


def native_crops(coarse, rectangles, adjustment=None, scale=3., floor=64 / 224):
    offset, extent = rectangles[:, :2], rectangles[:, 2:]
    if (extent <= 0).any():
        raise ValueError('Invalid native-to-canvas geometry')
    center = (coarse[:, :2] - offset) / extent
    size = torch.maximum(coarse[:, 2:] * scale, coarse.new_full(coarse[:, 2:].shape, floor)) / extent
    size = size.clamp(max=1.)
    center, size = center[:, None].expand(-1, 2, -1), size[:, None].expand(-1, 2, -1)
    if adjustment is not None:
        bounded = adjustment.tanh()
        center = center + .25 * size * bounded[..., :2]
        size = (size * (.35 * bounded[..., 2:]).exp()).clamp(max=1.)
    center = torch.maximum(size / 2, torch.minimum(center, 1 - size / 2))
    native = torch.cat((center, size), -1)
    canvas = torch.cat((center * extent[:, None] + offset[:, None], size * extent[:, None]), -1)
    return native, canvas


def sample_native(native, sizes, crops, modality, output_size=224):
    if native.ndim != 4 or native.shape[1] != 4 or modality not in (0, 1):
        raise ValueError('Expected a packed RGB-T native image batch')
    axis = (torch.arange(output_size, device=native.device, dtype=torch.float32) + .5) / output_size - .5
    y, x = torch.meshgrid(axis, axis, indexing='ij')
    unit = torch.stack((x, y), -1)
    grid = crops[:, None, None, :2] + unit * crops[:, None, None, 2:]
    padded = native.new_tensor([native.shape[-1], native.shape[-2]], dtype=torch.float32)
    dimensions = sizes[:, None, None, [1, 0]]
    pixels = grid * dimensions
    pixels = torch.maximum(pixels.new_tensor(.5), torch.minimum(pixels, dimensions - .5))
    grid = 2 * pixels / padded - 1
    channels = native[:, :3] if modality == 0 else native[:, 3:4]
    image = F.grid_sample(channels.float() / 255., grid.float(), mode='bilinear',
                          padding_mode='border', align_corners=False)
    if modality == 0:
        mean = image.new_tensor([.631, .6401, .632])[None, :, None, None]
        std = image.new_tensor([.2152, .227, .2439])[None, :, None, None]
        return (image - mean) / std
    return ((image - .5337) / .2562).expand(-1, 3, -1, -1)


class ReobserveHead(nn.Module):
    def __init__(self, dim=512, hidden=128):
        super().__init__()
        self.global_norm, self.visual_norm, self.text_norm = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.global_proj, self.visual_proj, self.text_proj = nn.Linear(dim, hidden), nn.Linear(dim, hidden), nn.Linear(dim, hidden)
        self.box_proj = nn.Linear(4, hidden)
        self.modality = nn.Parameter(torch.randn(2, hidden) * .02)
        self.language_read = nn.MultiheadAttention(hidden, 4, dropout=0., batch_first=True)
        self.policy = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 4))
        nn.init.zeros_(self.policy[-1].weight)
        nn.init.zeros_(self.policy[-1].bias)
        self.position = nn.Sequential(nn.Linear(2, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.local_read = nn.MultiheadAttention(hidden, 4, dropout=0., batch_first=True)
        self.output_norm = nn.LayerNorm(hidden)
        self.delta = nn.Linear(hidden, 4)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)
        self.blend = nn.Linear(2 * hidden, 2)
        self.accept = nn.Linear(2 * hidden, 1)
        nn.init.zeros_(self.blend.weight)
        nn.init.zeros_(self.blend.bias)
        nn.init.zeros_(self.accept.weight)
        nn.init.zeros_(self.accept.bias)

    def queries(self, reg, coarse, visual, text, text_padding):
        if text_padding.all(-1).any():
            raise ValueError('Text must contain at least one valid token')
        pooled = []
        # Read regional evidence on the original 14x14 grid only to decide
        # where to obtain new pixels. The final evidence comes from native crops.
        for tokens in visual:
            grid = (coarse[:, None, None, :2] * 2 - 1).float()
            feature = tokens[:, 1:].transpose(1, 2).reshape(-1, 512, 14, 14)
            pooled.append(F.grid_sample(feature, grid, align_corners=False).flatten(1))
        query = self.global_proj(self.global_norm(reg))[:, None] + self.box_proj(coarse)[:, None]
        query = query + self.visual_proj(self.visual_norm(torch.stack(pooled, 1))) + self.modality[None]
        language = self.text_proj(self.text_norm(text))
        context = self.language_read(query, language, language, key_padding_mask=text_padding, need_weights=False)[0]
        return query + context

    def decode(self, queries, coarse, canvas_crops, local):
        side = math.isqrt(local[0].shape[1] - 1)
        axis = (torch.arange(side, device=coarse.device, dtype=coarse.dtype) + .5) / side - .5
        y, x = torch.meshgrid(axis, axis, indexing='ij')
        unit = torch.stack((x.flatten(), y.flatten()), -1)
        contexts = []
        for modality, tokens in enumerate(local):
            crop = canvas_crops[:, modality]
            positions = crop[:, None, :2] + unit[None] * crop[:, None, 2:]
            relative = (positions - coarse[:, None, :2]) / coarse[:, None, 2:].clamp(min=1 / 224)
            memory = self.visual_proj(self.visual_norm(tokens[:, 1:])) + self.position(relative.clamp(-8, 8))
            context = self.local_read(queries[:, modality:modality + 1], memory, memory, need_weights=False)[0][:, 0]
            contexts.append(self.output_norm(queries[:, modality] + context))
        contexts = torch.stack(contexts, 1)
        raw = self.delta(contexts).tanh()
        center_delta = .5 * canvas_crops[..., 2:] * raw[..., :2]
        log_size_delta = math.log(2) * raw[..., 2:]
        proposals = torch.cat((coarse[:, None, :2] + center_delta,
                               coarse[:, None, 2:] * log_size_delta.exp()), -1)
        joined = contexts.flatten(1)
        weights = self.blend(joined).softmax(-1)
        accept_logits = self.accept(joined)[:, 0]
        strength = accept_logits.sigmoid()[:, None]
        center = coarse[:, :2] + strength * (weights[..., None] * center_delta).sum(1)
        size = coarse[:, 2:] * (strength * (weights[..., None] * log_size_delta).sum(1)).exp()
        final = torch.cat((center.clamp(0, 1), size.clamp(1e-5, 1)), -1)
        return final, {'reobserve_proposals': proposals, 'reobserve_accept_logits': accept_logits,
                       'reobserve_crops': canvas_crops, 'reobserve_coarse': coarse}


class LocalReobserveGrounder(nn.Module):
    def __init__(self, base, args):
        super().__init__()
        self.base, self.args = base, args
        self.mode = args.reobserve_mode
        self.refiner = None
        if self.mode != 'c1_continue':
            with torch.random.fork_rng(devices=[]):
                self.refiner = ReobserveHead(hidden=args.reobserve_hidden)
        self._cache, self._collect = {}, False
        def visual_hook(module, inputs, output):
            if self._collect:
                self._cache.setdefault('visual', []).append(output)
        def text_hook(module, inputs, output):
            if self._collect:
                self._cache['text_output'] = output
        def projection_hook(module, inputs, output):
            if self._collect:
                self._cache['text'] = output
        def vl_hook(module, inputs, output):
            if self._collect:
                self._cache['reg'] = output[0]
        self._hooks = [base.visu_proj.register_forward_hook(visual_hook),
                       base.clip.text_model.register_forward_hook(text_hook),
                       base.text_proj.register_forward_hook(projection_hook),
                       base.vl_transformer.register_forward_hook(vl_hook)]
        self.configure_tuning()

    def configure_tuning(self):
        prefixes = ('base.bbox_embed.', 'base.vl_transformer.encoder.layers.5.',
                    'base.vl_transformer.encoder.norm.', 'refiner.')
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name.startswith(prefixes))
        # PEFT changes requires_grad on every adapter switch. Restore the
        # explicit all-frozen visual mask through C1's existing switch hook.
        self.base._clip_control_trainability = {n: False for n, _ in self.base.clip.named_parameters()}

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        if self.refiner is not None:
            self.refiner.train(mode)
        return self

    def _encode_local(self, image, modality, cached_text):
        self.base._select_grounding_clip_adapter('lora_rgb' if modality == 0 else 'lora_ir')
        if self.base.mixup_pretrain:
            projected = self.base.clip.text_projection(cached_text.last_hidden_state)
            text_states = [self.base.condition_text_proj(projected.float())]
        else:
            text_states = [cached_text.hidden_states[i] for i in self.base.extract_text_layer]
        outputs = self.base.clip.vision_model(adapt_layer=self.base.adapt_layer, text_states=text_states,
            reg_src=self.base.reg_token.weight[None].expand(image.shape[0], -1, -1),
            cur_modality='rgb' if modality == 0 else 'ir', pixel_values=image,
            output_attentions=False, output_hidden_states=True, return_dict=True)
        levels = [outputs['hidden_states'][i] for i in self.base.extract_vision_layer]
        return self.base.visu_proj(self.base.ml_visual_projection(torch.cat(levels, -1)).float())

    def forward(self, img_data, text_data, target_boxes=None):
        images, phrases = img_data, text_data
        if target_boxes is not None:
            raise ValueError('GT boxes must not enter the re-observation forward path')
        self._cache, self._collect = {}, self.refiner is not None
        try:
            result = self.base(NestedTensor(images.tensors, images.mask), phrases)
        finally:
            self._collect = False
        if self.refiner is None:
            return result
        cache, self._cache = self._cache, {}
        coarse = result[0].detach()
        _, padding = self.base.encode_text(phrases, images.tensors.device)
        queries = self.refiner.queries(cache['reg'].detach(), coarse,
            [x.detach() for x in cache['visual'][:2]], cache['text'].detach(), padding)
        policy = self.refiner.policy(queries)
        adjustment = policy if self.mode in ('text_adaptive', 'text_consistent') else policy * 0.
        crops, canvas = native_crops(coarse, images.rectangles, adjustment)
        local = [self._encode_local(sample_native(images.native, images.sizes, crops[:, i], i), i,
                                    cache['text_output']) for i in range(2)]
        # Keep the coarse path trainable in every control while crop selection
        # remains detached from its geometry, so the reader cannot move C1's
        # coarse prediction just to obtain an easier crop.
        refined, aux = self.refiner.decode(queries, result[0], canvas, local)
        aux['reobserve_policy'] = policy
        main = (refined, *result[1:])
        return (*main, aux) if self.training else main
