import mlx.core as mx
import mlx.nn as nn

class SnakeBeta(nn.Module):
    def __init__(self, in_features: int, alpha: float = 1.0, alpha_logscale: bool = True):
        super().__init__()
        self.in_features = in_features
        self.alpha_logscale = alpha_logscale
        if alpha_logscale:
            self.alpha = mx.zeros((in_features,))
            self.beta = mx.zeros((in_features,))
        else:
            self.alpha = mx.ones((in_features,))
            self.beta = mx.ones((in_features,))

    def __call__(self, x: mx.array) -> mx.array:
        # x is [B, C, T]
        alpha = mx.exp(self.alpha)[None, :, None] if self.alpha_logscale else self.alpha[None, :, None]
        beta = mx.exp(self.beta)[None, :, None] if self.alpha_logscale else self.beta[None, :, None]
        return x + (1.0 / (beta + 1e-9)) * (mx.sin(x * alpha) ** 2)

class Activation1d(nn.Module):
    def __init__(self, activation):
        super().__init__()
        self.act = activation

    def __call__(self, x: mx.array) -> mx.array:
        # Direct periodic activation
        return self.act(x)
