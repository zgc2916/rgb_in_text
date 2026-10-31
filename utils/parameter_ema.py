"""Optimizer-step EMA over the explicitly selected trainable parameters."""
from contextlib import contextmanager

import torch


class ParameterEMA:
    def __init__(self, model, names, decay=.99):
        self.decay, self.updates = float(decay), 0
        parameters = dict(model.named_parameters())
        self.shadow = {name: parameters[name].detach().clone() for name in names}
        if not self.shadow or not 0 <= self.decay < 1:
            raise ValueError('EMA requires selected parameters and a decay in [0,1)')

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        decay = min(self.decay, (1 + self.updates) / (10 + self.updates))
        parameters = dict(model.named_parameters())
        for name, value in self.shadow.items():
            value.lerp_(parameters[name].detach(), 1 - decay)

    @contextmanager
    def apply(self, model):
        parameters = dict(model.named_parameters())
        raw = {name: parameters[name].detach().clone() for name in self.shadow}
        with torch.no_grad():
            for name, value in self.shadow.items():
                parameters[name].copy_(value)
        try:
            yield
        finally:
            with torch.no_grad():
                for name, value in raw.items():
                    parameters[name].copy_(value)

    def state_dict(self):
        return {'decay': self.decay, 'updates': self.updates,
                'shadow': {name: value.detach().cpu().clone() for name, value in self.shadow.items()}}

    def load_state_dict(self, state):
        if state['decay'] != self.decay or set(state['shadow']) != set(self.shadow):
            raise ValueError('EMA state belongs to a different tuning scope or decay')
        for name, value in state['shadow'].items():
            if value.shape != self.shadow[name].shape or value.dtype != self.shadow[name].dtype:
                raise ValueError('EMA tensor schema differs')
            self.shadow[name].copy_(value)
        self.updates = state['updates']
