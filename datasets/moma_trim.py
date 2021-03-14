import os
import torch
from torchvision import datasets, io
import torchvision.transforms.functional as F

from .moma_api import get_moma_api
import utils


class MOMATrim(datasets.VisionDataset):
  def __init__(self, cfg, fetch=()):
    super(MOMATrim, self).__init__(cfg.data_dir)

    self.cfg = cfg
    self.fetch = fetch
    self.api = get_moma_api(cfg.data_dir, 'trim')
    self.trim_ids = self.api.get_trim_ids()

    if 'feat' in self.fetch:
      self.feats = self.load_feats()

  @staticmethod
  def resize(video, trim_ann, scale=0.5):
    h, w = video.shape[-2:]
    video = F.resize(video, [round(h*scale), round(w*scale)])
    trim_ann['ags'] = [ag.resize(scale) for ag in trim_ann['ags']]

    return video, trim_ann

  def load_feats(self, feats_fname='feats.pt', chunk_sizes_fname='chunk_sizes.pt'):
    feats = torch.load(os.path.join(self.api.feats_dir, feats_fname))
    chunk_sizes = torch.load(os.path.join(self.api.feats_dir, chunk_sizes_fname))
    feats = utils.split_vl(feats, chunk_sizes)
    return feats

  def __getitem__(self, index):
    trim_id = self.trim_ids[index]
    trim_ann = self.api.get_ann(trim_id)
    out = [trim_id, trim_ann]

    if 'video' in self.fetch:
      video_path = self.api.get_video_path(trim_id, True)
      video = io.read_video(video_path, pts_unit='sec')[0]
      video = video.float().permute(0, 3, 1, 2)/255.  # to_tensor

      # resize when width > 2000
      if video.shape[-1] > 2000:
        video, trim_ann = self.resize(video, trim_ann)
        out[1] = trim_ann

      out.append(video)
    elif 'feat' in self.fetch:
      feat = self.feats[index]
      out.append(feat)

    return tuple(out)

  def __len__(self):
    return len(self.api.trim_ids)
