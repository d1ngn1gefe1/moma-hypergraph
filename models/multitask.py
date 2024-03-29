import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .encoders import encoders
from .decoders import ActHead, SActHead, PSVLAActHead, PAVLAActHead, PSFLAActHead, ActorHead, RelatHead
from .layers import MLP, ActorPooling, FramePooling
import utils


DIM_NODE_ATTR = 512
DIM_EDGE_ATTR = 9
DIM_ORC_NODE_ATTR = 140
DIM_ORC_EDGE_ATTR = 23


class MultitaskModel(nn.Module):
  def __init__(self, cfg, dim=256):
    super(MultitaskModel, self).__init__()

    self.cfg = cfg

    dim_hidden = dim*2 if cfg.oracle else dim
    self.encoder = encoders[cfg.backbone](dim=dim_hidden)
    self.actor_pooling = ActorPooling()
    self.frame_pooling = FramePooling()
    self.act_head = ActHead(num_classes=cfg.num_act_classes, dim=dim_hidden)
    self.sact_head = SActHead(num_classes=cfg.num_sact_classes, dim=dim_hidden)
    self.psvl_aact_head = PSVLAActHead(num_classes=cfg.num_aact_classes, dim=dim_hidden)
    self.pavl_aact_head = PAVLAActHead(num_classes=cfg.num_aact_classes, dim=dim_hidden)
    self.psfl_aact_head = PSFLAActHead(num_classes=cfg.num_aact_classes, dim=dim_hidden)
    self.actor_head = ActorHead(num_classes=cfg.num_actor_classes, dim=dim_hidden)
    self.relat_head = RelatHead(num_classes=cfg.num_relat_classes, dim=dim_hidden)

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
    embed_actors = self.actor_pooling(embed, data.node_video_chunk_sizes, data.batch_actor)
    embed_frames = self.frame_pooling(embed, data.node_video_chunk_sizes, data.batch_frame)

    logits_act = self.act_head(embed, data.batch)
    logits_sact = self.sact_head(embed, data.batch)
    logits_psvl_aact = self.psvl_aact_head(embed, data.batch)
    logits_pavl_aact = self.pavl_aact_head(embed_actors)
    logits_psfl_aact = self.psfl_aact_head(embed_frames)
    logits_actor = self.actor_head(embed_actors)
    logits_relat = self.relat_head(data.edge_index, embed, data.hyperedge_chunk_sizes)

    loss_act = F.cross_entropy(logits_act, data.act_cids)*self.cfg.weights[0]
    loss_sact = F.cross_entropy(logits_sact, data.sact_cids)*self.cfg.weights[1]
    loss_psvl_aact = F.binary_cross_entropy_with_logits(logits_psvl_aact, data.psvl_aact_cids)*self.cfg.weights[2]
    loss_pavl_aact = F.binary_cross_entropy_with_logits(logits_pavl_aact, data.pavl_aact_cids)*self.cfg.weights[3]
    loss_psfl_aact = F.binary_cross_entropy_with_logits(logits_psfl_aact, data.psfl_aact_cids)*self.cfg.weights[4]
    loss_actor = F.cross_entropy(logits_actor, data.actor_cids)*self.cfg.weights[5]
    loss_relat = F.cross_entropy(logits_relat, data.hyperedge_label)*self.cfg.weights[6]

    acc_act = utils.get_acc(logits_act, data.act_cids)
    acc_sact = utils.get_acc(logits_sact, data.sact_cids)
    acc_psvl_aact = utils.get_acc(logits_psvl_aact, data.psvl_aact_cids)
    acc_pavl_aact = utils.get_acc(logits_pavl_aact, data.pavl_aact_cids)
    acc_psfl_aact = utils.get_acc(logits_psfl_aact, data.psfl_aact_cids)
    acc_actor = utils.get_acc(logits_actor, data.actor_cids)
    acc_relat = utils.get_acc(logits_relat, data.hyperedge_label)

    mAP_act = utils.get_mAP(logits_act, data.act_cids)
    mAP_sact = utils.get_mAP(logits_sact, data.sact_cids)
    mAP_psvl_aact = utils.get_mAP(logits_psvl_aact, data.psvl_aact_cids)
    mAP_pavl_aact = utils.get_mAP(logits_pavl_aact, data.pavl_aact_cids)
    mAP_psfl_aact = utils.get_mAP(logits_psfl_aact, data.psfl_aact_cids)
    mAP_actor = utils.get_mAP(logits_actor, data.actor_cids)
    mAP_relat = utils.get_mAP(logits_relat, data.hyperedge_label)

    loss = 0
    if 'act' in self.cfg.tasks:
      loss += loss_act
    if 'sact' in self.cfg.tasks:
      loss += loss_sact
    if 'psvl_aact' in self.cfg.tasks:
      loss += loss_psvl_aact
    if 'pavl_aact' in self.cfg.tasks:
      loss += loss_pavl_aact
    if 'psfl_aact' in self.cfg.tasks:
      loss += loss_psfl_aact
    if 'actor' in self.cfg.tasks:
      loss += loss_actor
    if 'relat' in self.cfg.tasks:
      loss += loss_relat

    stats = {'loss_act': (loss_act.item(), logits_act.shape[0]),
             'loss_sact': (loss_sact.item(), logits_sact.shape[0]),
             'loss_psvl_aact': (loss_psvl_aact.item(), logits_psvl_aact.shape[0]),
             'loss_pavl_aact': (loss_pavl_aact.item(), logits_pavl_aact.shape[0]),
             'loss_psfl_aact': (loss_psfl_aact.item(), logits_psfl_aact.shape[0]),
             'loss_actor': (loss_actor.item(), logits_actor.shape[0]),
             'loss_relat': (loss_relat.item(), logits_relat.shape[0]),

             'acc_act': (acc_act, logits_act.shape[0]),
             'acc_sact': (acc_sact, logits_sact.shape[0]),
             'acc_psvl_aact': (acc_psvl_aact, logits_psvl_aact.shape[0]),
             'acc_pavl_aact': (acc_pavl_aact, logits_pavl_aact.shape[0]),
             'acc_psfl_aact': (acc_psfl_aact, logits_psfl_aact.shape[0]),
             'acc_actor': (acc_actor, logits_actor.shape[0]),
             'acc_relat': (acc_relat, logits_relat.shape[0]),

             'mAP_act': (mAP_act, logits_act.shape[0]),
             'mAP_sact': (mAP_sact, logits_sact.shape[0]),
             'mAP_psvl_aact': (mAP_psvl_aact, logits_psvl_aact.shape[0]),
             'mAP_pavl_aact': (mAP_pavl_aact, logits_pavl_aact.shape[0]),
             'mAP_psfl_aact': (mAP_psfl_aact, logits_psfl_aact.shape[0]),
             'mAP_actor': (mAP_actor, logits_actor.shape[0]),
             'mAP_relat': (mAP_relat, logits_relat.shape[0]),
             }

    return loss, stats
