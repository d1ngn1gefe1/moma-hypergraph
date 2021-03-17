import os
import torch
from torchvision import datasets, io
import torchvision.transforms.functional as F

from .moma_api import get_moma_api
import utils


class MOMATrim(datasets.VisionDataset):
  def __init__(self, cfg, split=None, fetch=None, level='trim'):
    super(MOMATrim, self).__init__(cfg.data_dir)

    self.cfg = cfg
    self.fetch = fetch
    self.api = get_moma_api(cfg.data_dir, cfg.split_by, cfg.feats_dname, level)
    self.trim_ids = self.api.get_trim_ids(split=split)

    if self.fetch == 'pyg':
      self.feats = self.load_feats()

    if split == 'train':
      self.add_cfg()

  def add_cfg(self):
    setattr(self.cfg, 'num_act_classes', len(self.api.act_cnames))
    setattr(self.cfg, 'num_sact_classes', len(self.api.sact_cnames))
    setattr(self.cfg, 'num_aact_classes', len(self.api.aact_cnames))
    setattr(self.cfg, 'num_actor_classes', len(self.api.actor_cnames))

    if self.fetch == 'pyg':
      setattr(self.cfg, 'num_feats', self.feats[0].shape[1])

  @staticmethod
  def resize(video, trim_ann, scale=0.5):
    h, w = video.shape[-2:]
    video = F.resize(video, [round(h*scale), round(w*scale)])
    trim_ann['ags'] = [ag.resize(scale) for ag in trim_ann['ags']]
    trim_ann['size'] = trim_ann['size'].resize(scale)

    return video, trim_ann

  def load_feats(self, feats_fname='feats.pt', chunk_sizes_fname='chunk_sizes.pt', trim_ids_fname='trim_ids.txt'):
    all_feats = torch.load(os.path.join(self.api.feats_dir, feats_fname))
    chunk_sizes = torch.load(os.path.join(self.api.feats_dir, chunk_sizes_fname))
    with open(os.path.join(self.api.feats_dir, trim_ids_fname), 'r') as f:
      all_trim_ids = f.read().splitlines()
    all_feats = utils.split_vl(all_feats, chunk_sizes)

    indices = [all_trim_ids.index(trim_id) for trim_id in self.trim_ids]
    feats = [all_feats[index] for index in indices]

    return feats

  def __getitem__(self, index):
    trim_id = self.trim_ids[index]
    trim_ann = self.api.get_ann(trim_id)

    if self.fetch == 'video':
      video_path = self.api.get_video_path(trim_id, True)
      video = io.read_video(video_path, pts_unit='sec')[0]
      video = video.float().permute(0, 3, 1, 2)/255.  # to_tensor

      # reduce size when width > 2000
      if video.shape[-1] > 2000:
        video, trim_ann = self.resize(video, trim_ann)

      return trim_id, trim_ann, video

    elif self.fetch == 'pyg':
      feat = self.feats[index]
      untrim_id = self.api.untrim_ids[trim_id]
      act_cid = self.api.act_cids[untrim_id]
      sact_cid = self.api.sact_cids[trim_id]
      data = utils.to_pyg_data(trim_ann, feat, act_cid, sact_cid, *self.cfg.oracles)

      return data

    else:
      return trim_id, trim_ann

  def __len__(self):
    return len(self.trim_ids)
