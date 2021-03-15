import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GINConv, global_add_pool
from torch_scatter import scatter_mean

import utils


class GIN(nn.Module):
  def __init__(self, num_feats, num_classes, dim=256):
    super(GIN, self).__init__()

    nn1 = nn.Sequential(nn.Linear(num_feats, dim), nn.ReLU(), nn.Linear(dim, dim))
    self.conv1 = GINConv(nn1)
    self.bn1 = nn.BatchNorm1d(dim)

    nn2 = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
    self.conv2 = GINConv(nn2)
    self.bn2 = nn.BatchNorm1d(dim)

    nn3 = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
    self.conv3 = GINConv(nn3)
    self.bn3 = nn.BatchNorm1d(dim)

    self.fc1 = nn.Linear(dim, dim)
    self.fc2 = nn.Linear(dim, num_classes)

  def forward(self, x, edge_index, batch):
    x = F.relu(self.conv1(x, edge_index))
    x = self.bn1(x)
    x = F.relu(self.conv2(x, edge_index))
    x = self.bn2(x)
    x = F.relu(self.conv3(x, edge_index))
    x = self.bn3(x)
    x = global_add_pool(x, batch)
    x = F.relu(self.fc1(x))
    x = F.dropout(x, p=0.5, training=self.training)
    x = self.fc2(x)
    return x


class GINModel(nn.Module):
  def __init__(self, cfg):
    super(GINModel, self).__init__()

    self.cfg = cfg
    self.net = GIN(num_feats=self.cfg.num_feats, num_classes=self.cfg.num_classes)

  def get_optimizer(self):
    optimizer = optim.Adam(self.net.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
    return optimizer

  def get_scheduler(self, optimizer):
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.cfg.num_epochs)
    return scheduler

  def forward(self, data):
    logits = self.net.forward(data.x, data.edge_index, data.batch)
    loss = F.cross_entropy(logits, data.y)
    acc = utils.accuracy(logits, data.y)

    return loss, acc

