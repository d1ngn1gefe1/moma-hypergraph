import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .encoders import encoders
from .decoders import ActHead, SActHead, PSAActHead, PAAActHead, ActorHead
from .layers import ActorPooling, MLP
import utils


DIM_NODE_ATTR = 512
DIM_EDGE_ATTR = 9
DIM_ORC_NODE_ATTR = 140
DIM_ORC_EDGE_ATTR = 23


class MultitaskModel(nn.Module):
  def __init__(self, cfg, dim=256):
    super(MultitaskModel, self).__init__()

    self.cfg = cfg
    self.encoder = encoders[cfg.backbone](dim=dim*2 if cfg.oracle else dim)
    self.actor_pooling = ActorPooling()
    self.act_head = ActHead(num_classes=cfg.num_act_classes, dim=dim*2 if cfg.oracle else dim)
    self.sact_head = SActHead(num_classes=cfg.num_sact_classes, dim=dim*2 if cfg.oracle else dim)
    self.ps_aact_head = PSAActHead(num_classes=cfg.num_aact_classes, dim=dim*2 if cfg.oracle else dim)
    self.pa_aact_head = PAAActHead(num_classes=cfg.num_aact_classes, dim=dim*2 if cfg.oracle else dim)
    self.actor_head = ActorHead(num_classes=cfg.num_actor_classes, dim=dim*2 if cfg.oracle else dim)

    self.mlp_node = MLP(DIM_NODE_ATTR, dim)
    self.mlp_edge = MLP(DIM_EDGE_ATTR, dim)
    if cfg.oracle:
      self.mlp_orc_node = MLP(DIM_ORC_NODE_ATTR, dim)
      self.mlp_orc_edge = MLP(DIM_ORC_EDGE_ATTR, dim)

    self.count_parameters()

  def count_parameters(self):
    num_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
    print(self)
    print('Number of trainable parameters: {}'.format(num_parameters))

  def get_optimizer(self):
    optimizer = optim.Adam(self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
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
    node_attr = self.mlp_node(data.x)
    edge_attr = self.mlp_edge(data.edge_attr)

    if self.cfg.oracle:
      orc_node_attr = self.mlp_orc_node(data.orc_node_attr)
      orc_edge_attr = self.mlp_orc_edge(data.orc_edge_attr)
      node_attr = torch.cat([node_attr, orc_node_attr], dim=-1)
      edge_attr = torch.cat([edge_attr, orc_edge_attr], dim=-1)

    embed = self.encoder(data.edge_index, node_attr, edge_attr)
    embed_actors = self.actor_pooling(embed, data.chunk_sizes, data.batch_actor)

    logits_act = self.act_head(embed, data.batch)
    logits_sact = self.sact_head(embed, data.batch)
    logits_ps_aact = self.ps_aact_head(embed, data.batch)
    logits_pa_aact = self.pa_aact_head(embed_actors)
    logits_actor = self.actor_head(embed_actors)

    loss_act = F.cross_entropy(logits_act, data.act_cids)*self.cfg.weights[0]
    loss_sact = F.cross_entropy(logits_sact, data.sact_cids)*self.cfg.weights[1]
    loss_ps_aact = F.binary_cross_entropy_with_logits(logits_ps_aact, data.ps_aact_cids)*self.cfg.weights[2]
    loss_pa_aact = F.binary_cross_entropy_with_logits(logits_pa_aact, data.pa_aact_cids)*self.cfg.weights[3]
    loss_actor = F.cross_entropy(logits_actor, data.actor_cids)*self.cfg.weights[4]

    acc_act = utils.get_acc(logits_act, data.act_cids)
    acc_sact = utils.get_acc(logits_sact, data.sact_cids)
    acc_ps_aact = utils.get_acc(logits_ps_aact, data.ps_aact_cids)
    acc_pa_aact = utils.get_acc(logits_pa_aact, data.pa_aact_cids)
    acc_actor = utils.get_acc(logits_actor, data.actor_cids)

    mAP_act = utils.get_mAP(logits_act, data.act_cids)
    mAP_sact = utils.get_mAP(logits_sact, data.sact_cids)
    mAP_ps_aact = utils.get_mAP(logits_ps_aact, data.ps_aact_cids)
    mAP_pa_aact = utils.get_mAP(logits_pa_aact, data.pa_aact_cids)
    mAP_actor = utils.get_mAP(logits_actor, data.actor_cids)

    loss = 0
    if 'act' in self.cfg.tasks:
      loss += loss_act
    if 'sact' in self.cfg.tasks:
      loss += loss_sact
    if 'ps_aact' in self.cfg.tasks:
      loss += loss_ps_aact
    if 'pa_aact' in self.cfg.tasks:
      loss += loss_pa_aact
    if 'actor' in self.cfg.tasks:
      loss += loss_actor

    stats = {'loss_act': loss_act.item(),
             'loss_sact': loss_sact.item(),
             'loss_ps_aact': loss_ps_aact.item(),
             'loss_pa_aact': loss_pa_aact.item(),
             'loss_actor': loss_actor.item(),

             'acc_act': acc_act,
             'acc_sact': acc_sact,
             'acc_ps_aact': acc_ps_aact,
             'acc_pa_aact': acc_pa_aact,
             'acc_actor': acc_actor,

             'mAP_act': mAP_act,
             'mAP_sact': mAP_sact,
             'mAP_ps_aact': mAP_ps_aact,
             'mAP_pa_aact': mAP_pa_aact,
             'mAP_actor': mAP_actor}

    return loss, stats
