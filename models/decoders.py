import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool

import utils


class ActHead(nn.Module):
  """ Activity classification
  """
  def __init__(self, num_classes, dim):
    super(ActHead, self).__init__()

    self.fc1 = nn.Linear(dim, dim)
    self.fc2 = nn.Linear(dim, num_classes)

  def forward(self, embed, batch_video):
    x = global_mean_pool(embed, batch_video)
    x = F.relu(self.fc1(x))
    x = F.dropout(x, p=0.5, training=self.training)
    x = self.fc2(x)
    return x


class SActHead(nn.Module):
  """ Sub-activity classification
  """
  def __init__(self, num_classes, dim):
    super(SActHead, self).__init__()

    self.fc1 = nn.Linear(dim, dim)
    self.fc2 = nn.Linear(dim, num_classes)

  def forward(self, embed, batch_video):
    x = global_mean_pool(embed, batch_video)
    x = F.relu(self.fc1(x))
    x = F.dropout(x, p=0.5, training=self.training)
    x = self.fc2(x)
    return x


class PSVLAActHead(nn.Module):
  """ Per-scene, video-level multi-label atomic action classification
  """
  def __init__(self, num_classes, dim):
    super(PSVLAActHead, self).__init__()

    self.fc1 = nn.Linear(dim, dim)
    self.fc2 = nn.Linear(dim, num_classes)

  def forward(self, embed, batch_video):
    x = global_mean_pool(embed, batch_video)
    x = F.relu(self.fc1(x))
    x = F.dropout(x, p=0.5, training=self.training)
    x = self.fc2(x)
    return x


class PAVLAActHead(nn.Module):
  """ Per-actor, video-level multi-label atomic action classification
  """
  def __init__(self, num_classes, dim):
    super(PAVLAActHead, self).__init__()

    self.fc1 = nn.Linear(dim, dim)
    self.fc2 = nn.Linear(dim, num_classes)

  def forward(self, embed_actors):
    x = F.relu(self.fc1(embed_actors))
    x = F.dropout(x, p=0.5, training=self.training)
    x = self.fc2(x)

    return x


class PSFLAActHead(nn.Module):
  """ Per-scene, frame-level multi-label atomic action classification
  """
  def __init__(self, num_classes, dim):
    super(PSFLAActHead, self).__init__()

    self.fc1 = nn.Linear(dim, dim)
    self.fc2 = nn.Linear(dim, num_classes)

  def forward(self, embed):
    x = F.relu(self.fc1(embed))
    x = F.dropout(x, p=0.5, training=self.training)
    x = self.fc2(x)
    return x


class ActorHead(nn.Module):
  """ Actor role classification
  """
  def __init__(self, num_classes, dim):
    super(ActorHead, self).__init__()

    self.fc1 = nn.Linear(dim, dim)
    self.fc2 = nn.Linear(dim, num_classes)

  def forward(self, embed_actors):
    x = F.relu(self.fc1(embed_actors))
    x = F.dropout(x, p=0.5, training=self.training)
    x = self.fc2(x)

    return x


class RelatHead(nn.Module):
  """ Relationship classification
  """
  def __init__(self, num_classes, dim):
    super(RelatHead, self).__init__()

    self.fc1 = nn.Linear(dim*2, dim)
    self.fc2 = nn.Linear(dim, num_classes)

  def forward(self, edge_index, embed, hyperedge_chunk_sizes):
    if edge_index.shape[1] == 0:  # no edge: will trigger bug in global_mean_pool
      return 0

    batch_relat = utils.to_batch(hyperedge_chunk_sizes)

    embed_src = embed[edge_index[0]]
    embed_snk = embed[edge_index[1]]
    embed_edge = torch.cat((embed_src, embed_snk), dim=-1)

    x = F.relu(self.fc1(embed_edge))
    x = F.dropout(x, p=0.5, training=self.training)
    x = self.fc2(x)

    x = global_mean_pool(x, batch_relat)

    return x
