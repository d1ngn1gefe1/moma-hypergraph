import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, HypergraphConv


class GINEncoder(nn.Module):
  def __init__(self, num_feats, dim=256):
    super(GINEncoder, self).__init__()

    nn1 = nn.Sequential(nn.Linear(num_feats, dim), nn.ReLU(), nn.Linear(dim, dim))
    self.conv1 = GINConv(nn1)
    self.bn1 = nn.BatchNorm1d(dim)

    nn2 = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
    self.conv2 = GINConv(nn2)
    self.bn2 = nn.BatchNorm1d(dim)

    nn3 = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
    self.conv3 = GINConv(nn3)
    self.bn3 = nn.BatchNorm1d(dim)

  def forward(self, x, edge_index):
    x = F.relu(self.conv1(x, edge_index))
    x = self.bn1(x)
    x = F.relu(self.conv2(x, edge_index))
    x = self.bn2(x)
    x = F.relu(self.conv3(x, edge_index))
    x = self.bn3(x)
    return x


class HGCNEncoder(nn.Module):
  def __init__(self, num_feats, dim=256):
    super(HGCNEncoder, self).__init__()

    self.conv1 = HypergraphConv(num_feats, dim)
    self.nn1 = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
    self.bn1 = nn.BatchNorm1d(dim)

    self.conv2 = HypergraphConv(dim, dim)
    self.nn2 = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
    self.bn2 = nn.BatchNorm1d(dim)

    self.conv3 = HypergraphConv(dim, dim)
    self.nn3 = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
    self.bn3 = nn.BatchNorm1d(dim)

  def forward(self, x, edge_index):
    x = self.conv1(x, edge_index)
    x = F.relu(self.nn1(x))
    x = self.bn1(x)
    x = self.conv2(x, edge_index)
    x = F.relu(self.nn2(x))
    x = self.bn2(x)
    x = self.conv3(x, edge_index)
    x = F.relu(self.nn3(x))
    x = self.bn3(x)
    return x


encoders = {'GIN': GINEncoder, 'HGCN': HGCNEncoder}
