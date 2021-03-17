import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool

import utils


class MLP(nn.Module):
  def __init__(self, in_dim, out_dim):
    super(MLP, self).__init__()

    self.linear = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU())
    self.bn = nn.BatchNorm1d(out_dim)

  def forward(self, x):
    x = self.linear(x)
    x = self.bn(x)

    return x


class ActorPooling(nn.Module):
  def __init__(self):
    super(ActorPooling, self).__init__()

  def forward(self, x, chunk_sizes, batch_actor):
    batch_actor_list = utils.split_vl(batch_actor, chunk_sizes)
    embed_list = utils.split_vl(x, chunk_sizes)

    # loop across videos
    embed_actors = []
    for batch_actor, embed in zip(batch_actor_list, embed_list):
      embed_actor = global_mean_pool(embed, batch_actor)[1:]  # skip first: object
      embed_actors.append(embed_actor)
    embed_actors = torch.cat(embed_actors, dim=0)

    return embed_actors
