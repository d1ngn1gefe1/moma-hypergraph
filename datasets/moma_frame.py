import numpy as np
import os
import torch
from torch_geometric.data import Dataset, InMemoryDataset, Data

from .moma_api import get_moma_api


class MOMAFrame(InMemoryDataset):
  def __init__(self, cfg, split):
    self.cfg = cfg
    self.split = split
    self.api = get_moma_api(cfg.data_dir, 'frame')

    cfg.num_actor_classes = len(self.api.actor_cnames)
    cfg.num_object_classes = len(self.api.object_cnames)
    cfg.num_relat_classes = len(self.api.relat_cnames)

    super(MOMAFrame, self).__init__(cfg.data_dir)

    self.data, self.slices = torch.load(self.processed_paths[0])

  @property
  def processed_dir(self):
    return os.path.join(self.root, 'processed')

  @property
  def processed_file_names(self):
    return 'dataset_{}.pt'.format(self.split)

  def process(self):
    indices_train = np.load(os.path.join(self.cfg.data_dir, 'split_spatial_graph.npy'))
    keys = list(self.api.annotations.keys())

    if self.split == 'train':
      indices = indices_train
    else:
      assert self.split == 'val'
      indices = np.setdiff1d(np.arange(len(keys)), indices_train)
      assert len(keys) == len(indices_train)+len(indices)

    keys = [keys[index] for index in indices]

    data_list = []
    for key in keys:
      annotation = self.api.annotations[key]
      data = to_data(annotation['graph'], annotation['subactivity_cid'],
                     self.cfg.num_actor_classes, self.cfg.num_object_classes, self.cfg.num_relat_classes)
      data_list.append(data)

    data, slices = self.collate(data_list)
    torch.save((data, slices), self.processed_paths[0])
