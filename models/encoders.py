import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, GINConv, HypergraphConv


class GINEEncoder(nn.Module):
  def __init__(self, dim):
    super(GINEEncoder, self).__init__()

    self.conv1 = GINEConv(nn.Linear(dim, dim))
    self.bn1 = nn.BatchNorm1d(dim)

    self.conv2 = GINEConv(nn.Linear(dim, dim))
    self.bn2 = nn.BatchNorm1d(dim)

    self.conv3 = GINEConv(nn.Linear(dim, dim))
    self.bn3 = nn.BatchNorm1d(dim)

  def forward(self, edge_index, node_attr, edge_attr):
    x = F.relu(self.conv1(node_attr, edge_index, edge_attr))
    x = self.bn1(x)
    x = F.relu(self.conv2(x, edge_index, edge_attr))
    x = self.bn2(x)
    x = F.relu(self.conv3(x, edge_index, edge_attr))
    x = self.bn3(x)
    return x


class GINEncoder(nn.Module):
  def __init__(self, dim):
    super(GINEncoder, self).__init__()

    self.conv1 = GINConv(nn.Linear(dim, dim))
    self.bn1 = nn.BatchNorm1d(dim)

    self.conv2 = GINConv(nn.Linear(dim, dim))
    self.bn2 = nn.BatchNorm1d(dim)

    self.conv3 = GINConv(nn.Linear(dim, dim))
    self.bn3 = nn.BatchNorm1d(dim)

  def forward(self, edge_index, node_attr, edge_attr):
    x = F.relu(self.conv1(node_attr, edge_index))
    x = self.bn1(x)
    x = F.relu(self.conv2(x, edge_index))
    x = self.bn2(x)
    x = F.relu(self.conv3(x, edge_index))
    x = self.bn3(x)
    return x


class HGCNEncoder(nn.Module):
  def __init__(self, dim):
    super(HGCNEncoder, self).__init__()

    self.conv1 = HypergraphConv(dim, dim)
    self.bn1 = nn.BatchNorm1d(dim)

    self.conv2 = HypergraphConv(dim, dim)
    self.bn2 = nn.BatchNorm1d(dim)

    self.conv3 = HypergraphConv(dim, dim)
    self.bn3 = nn.BatchNorm1d(dim)

  def forward(self, edge_index, node_attr, edge_attr):
    x = F.relu(self.conv1(node_attr, edge_index))
    x = self.bn1(x)
    x = F.relu(self.conv2(x, edge_index))
    x = self.bn2(x)
    x = F.relu(self.conv3(x, edge_index))
    x = self.bn3(x)
    return x


encoders = {'GINE': GINEEncoder, 'GIN': GINEncoder, 'HGCN': HGCNEncoder}
