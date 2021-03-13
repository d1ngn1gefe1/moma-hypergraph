import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv
from torch_scatter import scatter_mean

import utils


class RGCN(nn.Module):
  def __init__(self, num_node_features, num_relations, num_classes):
    super(RGCN, self).__init__()
    self.conv1 = RGCNConv(num_node_features, 16, num_relations)
    self.conv2 = RGCNConv(16, num_classes, num_relations)

  def forward(self, data):
    x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

    x = self.conv1(x, edge_index, edge_attr)
    x = F.relu(x)
    x = F.dropout(x, training=self.training)
    x = self.conv2(x, edge_index, edge_attr)

    logits = scatter_mean(x, data.batch, dim=0)

    return logits


class RGCNModel(nn.Module):
  def __init__(self, cfg):
    super(RGCNModel, self).__init__()

    #todo
    cfg.num_actor_classes = 6
    cfg.num_object_classes = 7
    cfg.num_relat_classes = 8
    cfg.num_classes = 9

    self.cfg = cfg
    self.net = RGCN(cfg.num_actor_classes+cfg.num_object_classes, cfg.num_relat_classes, cfg.num_classes)

  def get_optimizer(self):
    optimizer = optim.Adam(self.net.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
    return optimizer

  def forward(self, data):
    return self.net.forward(data)

  def step(self, data):
    logits = self.forward(data)
    loss = F.cross_entropy(logits, data.y)
    acc = utils.accuracy(logits, data.y)

    return loss, acc

