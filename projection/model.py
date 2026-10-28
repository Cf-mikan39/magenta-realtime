"""Only this small MLP is trained; no teacher/image model is imported."""

import torch
from torch import nn
from torch.nn import functional as F


class ProjectionMLP(nn.Module):
  def __init__(self, input_dim, hidden_dim=1024, activation="gelu", dropout=.1,
               layer_norm=False, output_dim=768):
    super().__init__()
    if input_dim < 1 or hidden_dim < 1 or output_dim != 768 or not 0 <= dropout < 1:
      raise ValueError("Invalid MLP dimensions/dropout; output_dim must be 768")
    activations = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}
    if activation not in activations:
      raise ValueError(f"Unsupported activation: {activation}")
    layers = [nn.Linear(input_dim, hidden_dim)]
    if layer_norm:
      layers.append(nn.LayerNorm(hidden_dim))
    layers += [activations[activation](), nn.Dropout(dropout), nn.Linear(hidden_dim, output_dim)]
    self.layers = nn.Sequential(*layers)

  def forward(self, values):
    return self.layers(values)


def normalized_output(values, normalization):
  if normalization == "l2":
    return F.normalize(values, dim=-1, eps=1e-8)
  if normalization != "none":
    raise ValueError("Training output normalization must be none or l2")
  return values


def loss_terms(predicted, target, config):
  mse = (predicted - target).square().mean(dim=-1)
  cosine = 1 - F.cosine_similarity(predicted, target, dim=-1, eps=1e-8)
  norm = (predicted.norm(dim=-1) - target.norm(dim=-1)).square()
  total = config["mse_weight"] * mse + config["cosine_weight"] * cosine + config["norm_weight"] * norm
  return total, mse, cosine, norm
