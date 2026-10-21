from collections.abc import Iterable

import torch
from torch import nn


def _build_mlp(layer_dims: list[int]) -> nn.Sequential:
    """按给定维度构造 MLP，最后一个 Linear 后面不接激活函数。"""

    layers = []
    for index in range(len(layer_dims) - 1):
        layers.append(nn.Linear(layer_dims[index], layer_dims[index + 1]))
        if index != len(layer_dims) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class AE(nn.Module):
    """经典全连接自编码器：对称编码器/解码器，隐藏层 ReLU，线性输出。"""

    def __init__(self, input_dim: int, hidden_dims: Iterable[int], latent_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = list(hidden_dims)
        self.latent_dim = latent_dim

        self.encoder = _build_mlp([input_dim, *self.hidden_dims, latent_dim])
        self.decoder = _build_mlp([latent_dim, *reversed(self.hidden_dims), input_dim])

    def forward(self, x: torch.Tensor) -> dict:
        """返回训练和打分都需要的隐空间表示与重构结果。"""

        z = self.encoder(x)
        return {
            "latent": z,
            "recon": self.decoder(z),
        }
