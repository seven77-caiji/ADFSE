import torch.nn.functional as F


def reconstruction_loss(target, reconstruction):
    """对数据或 [数据, 位置] 的所有维度等权计算 MSE。"""
    return F.mse_loss(reconstruction, target)
