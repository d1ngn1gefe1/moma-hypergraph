"""
Extract actor and object features
"""
import argparse
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.ops as ops
import torchvision.transforms as transforms

import datasets
import utils


class FeatExtractorModel(nn.Module):
  def __init__(self):
    super(FeatExtractorModel, self).__init__()
    self.net = models.resnet18(pretrained=True)
    self.net.layer4.register_forward_hook(self.hook_fn)
    self.buffer = {}

  def hook_fn(self, module, input, output):
    self.buffer[input[0].device] = output

  def forward(self, video):
    self.net(video)
    return self.buffer[video.device]


class FeatExtractor:
  def __init__(self, cfg):
    self.cfg = cfg
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  @staticmethod
  def extract_bboxes(ags):
    bboxes = []
    for i, ag in enumerate(ags):
      for actor in ag.actors:
        x1, y1 = actor.bbox.x1, actor.bbox.y1
        x2, y2 = x1+actor.bbox.w, y1+actor.bbox.h
        bboxes.append([i, x1, y1, x2, y2])
      for object in ag.objects:
        x1, y1 = object.bbox.x1, object.bbox.y1
        x2, y2 = x1+object.bbox.w, y1+object.bbox.h
        bboxes.append([i, x1, y1, x2, y2])
    return bboxes

  def fit(self, model, dataset):
    feat_list, trim_ids = [], []

    model = model.to(self.device)
    model = nn.DataParallel(model)

    transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    dataloader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=self.cfg.num_workers)

    model.eval()
    with torch.no_grad():
      for i, (trim_id, trim_ann, video) in enumerate(dataloader):
        print('[{}] {}: len={}'.format(i, trim_id, video.shape[0]))
        assert video.shape[0] == len(trim_ann['ags'])

        bboxes = self.extract_bboxes(trim_ann['ags'])
        bboxes = torch.Tensor(bboxes).to(self.device)
        video = video.to(self.device)
        video = transform(video)

        # save memory
        if video.shape[0] > self.cfg.batch_size:
          print('split')
          num_steps = math.ceil(video.shape[0]/self.cfg.batch_size)
          feat = []
          for step in range(num_steps):
            start = step*self.cfg.batch_size
            end = min((step+1)*self.cfg.batch_size, video.shape[0])
            ret = model(video[start:end])
            feat.append(ret)
          feat = torch.cat(feat, dim=0)
        else:
          feat = model(video)

        feat = ops.roi_align(feat, bboxes, (7, 7), 1/32)
        feat = F.adaptive_avg_pool2d(feat, (1, 1))
        feat = torch.flatten(feat, 1)

        assert feat.shape[0] == bboxes.shape[0]
        feat_list.append(feat.detach().cpu())
        trim_ids.append(trim_id)

    feats, chunk_sizes = utils.cat_vl(feat_list)
    os.makedirs(dataset.api.feats_dir, exist_ok=True)
    torch.save(feats, os.path.join(dataset.api.feats_dir, 'feats.pt'))
    torch.save(chunk_sizes, os.path.join(dataset.api.feats_dir, 'chunk_sizes.pt'))
    with open(os.path.join(dataset.api.feats_dir, 'trim_ids.txt'), 'w+') as f:
      f.write('\n'.join(trim_ids))


parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', default='/home/ubuntu/datasets/MOMA', type=str)
parser.add_argument('--num_workers', default=32, type=int)
parser.add_argument('--batch_size', default=50, type=int)
parser.add_argument('--split_by', default='untrim', type=str, choices=['trim', 'untrim'])


def main():
  cfg = parser.parse_args()
  model = FeatExtractorModel()
  dataset = datasets.MOMATrim(cfg, split=None, fetch='video')
  feat_extractor = FeatExtractor(cfg)
  feat_extractor.fit(model, dataset)


if __name__ == '__main__':
  main()
