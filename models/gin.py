from itertools import chain
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GINConv, global_mean_pool

import utils


class Encoder(nn.Module):
  def __init__(self, num_feats, dim=256):
    super(Encoder, self).__init__()

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


class ActHead(nn.Module):
  def __init__(self, num_classes, dim=256):
    super(ActHead, self).__init__()

    self.fc1 = nn.Linear(dim, dim)
    self.fc2 = nn.Linear(dim, num_classes)

  def forward(self, x, batch_video):
    x = global_mean_pool(x, batch_video)
    x = F.relu(self.fc1(x))
    x = F.dropout(x, p=0.5, training=self.training)
    x = self.fc2(x)
    return x


class SActHead(nn.Module):
  def __init__(self, num_classes, dim=256):
    super(SActHead, self).__init__()

    self.fc1 = nn.Linear(dim, dim)
    self.fc2 = nn.Linear(dim, num_classes)

  def forward(self, x, batch_video):
    x = global_mean_pool(x, batch_video)
    x = F.relu(self.fc1(x))
    x = F.dropout(x, p=0.5, training=self.training)
    x = self.fc2(x)
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


class RoleHead(nn.Module):
  def __init__(self, num_classes, dim=256):
    super(RoleHead, self).__init__()

    self.fc1 = nn.Linear(dim, dim)
    self.fc2 = nn.Linear(dim, num_classes)

  def forward(self, embed_actors):
    x = F.relu(self.fc1(embed_actors))
    x = F.dropout(x, p=0.5, training=self.training)
    x = self.fc2(x)

    return x


class AActHead(nn.Module):
  def __init__(self, num_classes, dim=256):
    super(AActHead, self).__init__()

    self.fc1 = nn.Linear(dim, dim)
    self.fc2 = nn.Linear(dim, num_classes)

  def forward(self, embed_actors):
    x = F.relu(self.fc1(embed_actors))
    x = F.dropout(x, p=0.5, training=self.training)
    x = self.fc2(x)

    return x


class GINModel(nn.Module):
  def __init__(self, cfg):
    super(GINModel, self).__init__()

    self.cfg = cfg
    self.encoder = Encoder(num_feats=self.cfg.num_feats)
    self.actor_pooling = ActorPooling()
    self.act_head = ActHead(num_classes=self.cfg.num_act_classes)
    self.sact_head = SActHead(num_classes=self.cfg.num_sact_classes)
    # self.aact_head = RoleHead(num_classes=self.cfg.num_aact_classes)
    self.role_head = RoleHead(num_classes=self.cfg.num_actor_classes)

  def get_optimizer(self):
    parameters = [self.encoder.parameters(), self.act_head.parameters(), self.sact_head.parameters()]
    optimizer = optim.Adam(chain(*parameters), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
    return optimizer

  def get_scheduler(self, optimizer):
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.cfg.num_epochs)
    return scheduler

  def forward(self, data):
    """
     - batch_video (=data.batch): scatter batch feat by video
     - batch_frame: scatter per-video feat by frame
     - batch_actor: scatter per-video feat by frame
    """
    embed = self.encoder(data.x, data.edge_index)
    embed_actors = self.actor_pooling(embed, data.chunk_sizes, data.batch_actor)

    logits_act = self.act_head(embed, data.batch)
    logits_sact = self.sact_head(embed, data.batch)
    logits_role = self.role_head(embed_actors)

    loss_act = F.cross_entropy(logits_act, data.act_cids)
    loss_sact = F.cross_entropy(logits_sact, data.sact_cids)
    loss_role = F.cross_entropy(logits_role, data.actor_cids)
    acc_act = utils.accuracy(logits_act, data.act_cids)
    acc_sact = utils.accuracy(logits_sact, data.sact_cids)
    acc_role = utils.accuracy(logits_role, data.actor_cids)

    # loss = loss_act+loss_sact+loss_role
    loss = loss_role
    stats = {'loss_act': loss_act.item(),
             'loss_sact': loss_sact.item(),
             'loss_role': loss_role.item(),
             'acc_act': acc_act.item(),
             'acc_sact': acc_sact.item(),
             'acc_role': acc_role.item()}

    return loss, stats

