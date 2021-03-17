import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool


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


class PSAActHead(nn.Module):
  """ Per-scene, multi-label atomic action classification
  """
  def __init__(self, num_classes, dim):
    super(PSAActHead, self).__init__()

    self.fc1 = nn.Linear(dim, dim)
    self.fc2 = nn.Linear(dim, num_classes)

  def forward(self, embed, batch_video):
    x = global_mean_pool(embed, batch_video)
    x = F.relu(self.fc1(x))
    x = F.dropout(x, p=0.5, training=self.training)
    x = self.fc2(x)
    return x


class PAAActHead(nn.Module):
  """ Per-actor, multi-label atomic action classification
  """
  def __init__(self, num_classes, dim):
    super(PAAActHead, self).__init__()

    self.fc1 = nn.Linear(dim, dim)
    self.fc2 = nn.Linear(dim, num_classes)

  def forward(self, embed_actors):
    x = F.relu(self.fc1(embed_actors))
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
