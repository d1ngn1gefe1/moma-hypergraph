"""
Extract actor and object features
"""
import math
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import io
import torchvision.ops as ops
import torchvision.transforms as transforms

import utils


class FeatExtractor:
  def __init__(self, cfg):
    self.cfg = cfg
    self.device = torch.device('cuda:{}'.format(self.cfg.gpu) if torch.cuda.is_available() else 'cpu')
    self.buffer_feat = None

  @staticmethod
  def extract_bboxes(ahgs):
    bboxes = []
    for i, ahg in enumerate(ahgs):
      for actor in ahg.actors:
        x1, y1 = actor.bbox.x1, actor.bbox.y1
        x2, y2 = x1+actor.bbox.w, y1+actor.bbox.h
        bboxes.append([i, x1, y1, x2, y2])
      for object in ahg.objects:
        x1, y1 = object.bbox.x1, object.bbox.y1
        x2, y2 = x1+object.bbox.w, y1+object.bbox.h
        bboxes.append([i, x1, y1, x2, y2])
    return bboxes

  def hook_fn(self, module, input, output):
    self.buffer_feat = output

  def fit(self, model, dataset):
    feat_list = []

    model = model.to(self.device)
    model.layer4.register_forward_hook(self.hook_fn)

    transform = transforms.Compose([
      transforms.Lambda(lambda x: x.float().permute(0, 3, 1, 2)/255.),  # to_tensor
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataloader = DataLoader(dataset, batch_size=None, shuffle=False)

    model.eval()
    with torch.no_grad():
      for trm_id, trm_ann in dataloader:
        bboxes = self.extract_bboxes(trm_ann['ahgs'])
        bboxes = torch.Tensor(bboxes).to(self.device)

        video_path = dataset.api.get_video_path(trm_id, sample=True)
        video = io.read_video(video_path, pts_unit='sec')[0]
        video = video.to(self.device)
        video = transform(video)
        print('{}: {}'.format(trm_id, video.shape))
        assert video.shape[0] == len(trm_ann['ahgs'])

        # save memory
        feat = []
        steps = math.ceil(video.shape[0]/self.cfg.batch_size)
        for i in range(steps):
          s = i*self.cfg.batch_size
          e = min((i+1)*self.cfg.batch_size, video.shape[0])
          model(video[s:e])
          feat.append(self.buffer_feat)
        feat = torch.cat(feat, dim=0)

        feat = ops.roi_align(feat, bboxes, (7, 7), 1/32)
        feat = F.adaptive_avg_pool2d(feat, (1, 1))
        feat = torch.flatten(feat, 1)

        assert feat.shape[0] == bboxes.shape[0]
        feat_list.append(feat.detach().cpu())

    feats_dir = os.path.join(self.cfg.data_dir, 'feats')
    os.makedirs(feats_dir, exist_ok=True)
    feats, chunk_sizes = utils.cat_vl(feat_list)
    torch.save(feats, os.path.join(feats_dir, 'feats.pt'))
    torch.save(chunk_sizes, os.path.join(feats_dir, 'chunk_sizes.pt'))
