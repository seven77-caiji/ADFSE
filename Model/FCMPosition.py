"""FCM structure and position membership for anomaly detection."""

import numpy as np
import torch
from scipy.spatial.distance import cdist
from sklearn.cluster import kmeans_plusplus
from threadpoolctl import threadpool_limits
from torch import nn

from Model.AE import AE


def membership_from_distances(squared_distances, fuzzifier=1.3):
    """Compute standard FCM membership, splitting exact zero ties evenly."""
    if fuzzifier <= 1:
        raise ValueError("FCM fuzzifier must be greater than 1.")
    d2 = np.asarray(squared_distances, dtype=np.float64)
    if d2.ndim != 2 or np.any(d2 < 0) or not np.isfinite(d2).all():
        raise ValueError("Need a finite nonnegative 2D distance matrix.")
    zero = d2 <= 1e-24
    any_zero = zero.any(axis=1)
    result = np.empty_like(d2)
    if np.any(any_zero):
        rows = zero[any_zero]
        result[any_zero] = rows / rows.sum(axis=1, keepdims=True)
    if np.any(~any_zero):
        positive = d2[~any_zero]
        # Divide by each row minimum before exponentiation for stability.
        scaled = (positive / positive.min(axis=1, keepdims=True)) ** (-1.0 / (fuzzifier - 1.0))
        result[~any_zero] = scaled / scaled.sum(axis=1, keepdims=True)
    return result


def fit_fcm(x, n_clusters, seed=42, fuzzifier=1.3, max_iter=4000, tol=1e-4):
    """Fit FCM on normal training samples only; return centers and diagnostics."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or len(x) < n_clusters or n_clusters < 2 or not np.isfinite(x).all():
        raise ValueError("Need finite 2D training data and 2 <= n_clusters <= n_train.")
    if fuzzifier <= 1 or max_iter < 1 or tol <= 0:
        raise ValueError("Invalid FCM convergence parameters.")
    with threadpool_limits(limits=1):
        initial, _ = kmeans_plusplus(x, n_clusters=n_clusters, random_state=seed)
        centers = np.asarray(initial, dtype=np.float64)
        for iteration in range(1, max_iter + 1):
            d2 = cdist(x, centers, metric="sqeuclidean")
            membership = membership_from_distances(d2, fuzzifier)
            weights = membership ** fuzzifier
            mass = weights.sum(axis=0)
            if np.any(mass <= 0):
                raise FloatingPointError("FCM produced an empty cluster.")
            updated = (weights.T @ x) / mass[:, None]
            shift = np.max(np.linalg.norm(updated - centers, axis=1))
            centers = updated
            if shift <= tol:
                break
        final_d2 = cdist(x, centers, metric="sqeuclidean")
        final_membership = membership_from_distances(final_d2, fuzzifier)
        objective = float(np.sum(final_membership ** fuzzifier * final_d2))
    return {
        "centers": centers.astype(np.float32),
        "n_iter": iteration,
        "converged": shift <= tol,
        "objective": objective,
    }


class FCMPosition(nn.Module):
    """FCM membership to fixed training centers, including exact zero ties."""

    def __init__(self, centers, fuzzifier=1.3):
        super().__init__()
        if fuzzifier <= 1:
            raise ValueError("FCM fuzzifier must be greater than 1.")
        self.register_buffer("centers", torch.as_tensor(centers, dtype=torch.float32))
        self.fuzzifier = float(fuzzifier)

    def forward(self, x):
        d2 = (x[:, None, :] - self.centers[None, :, :]).square().sum(-1)
        zero = d2 <= 1e-24
        zero_membership = zero.to(d2.dtype) / zero.sum(dim=1, keepdim=True).clamp_min(1)
        smallest = d2.clamp_min(1e-24).amin(dim=1, keepdim=True)
        scaled = (d2.clamp_min(1e-24) / smallest).pow(-1.0 / (self.fuzzifier - 1.0))
        regular = scaled / scaled.sum(dim=1, keepdim=True)
        return torch.where(zero.any(dim=1, keepdim=True), zero_membership, regular)


class PositionAE(nn.Module):
    """Jointly reconstruct [features, FCM memberships]; score features only."""

    def __init__(self, input_dim, hidden_dims, latent_dim, centers, fuzzifier=1.3):
        super().__init__()
        self.input_dim = input_dim
        self.position = FCMPosition(centers, fuzzifier)
        self.ae = AE(input_dim + len(centers), hidden_dims, latent_dim)

    def forward(self, x):
        position = self.position(x)
        target = torch.cat([x, position], dim=1)
        out = self.ae(target)
        return {
            "latent": out["latent"],
            "target": target,
            "joint_recon": out["recon"],
            "recon": out["recon"][:, :self.input_dim],
            "position_recon": out["recon"][:, self.input_dim:],
        }
