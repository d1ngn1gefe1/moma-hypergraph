import numpy as np
import os
import torch
from torch_geometric.data import InMemoryDataset, Data

from .moma_api import get_momaapi


def to_data(graph, subactivity_cid, num_actor_classes, num_object_classes, num_relationship_classes):
  node_cids = [actor['actor_cid'] for actor in graph['actors']]+ \
              [object['object_cid']+num_actor_classes for object in graph['objects']]
  instance_ids = [actor['instance_id'] for actor in graph['actors']]+ \
                 [object['instance_id'] for object in graph['objects']]

  num_node_features = num_actor_classes+num_object_classes
  num_edge_features = num_relationship_classes

  x = np.zeros((len(node_cids), num_node_features), dtype=np.float32)
  x[np.arange(len(node_cids)), node_cids] = 1

  edges, edge_cids = [], []
  for relationship in graph['relationships']:
    source_node_instance_ids, sink_node_instance_ids = relationship['description'].split('),(')
    source_node_instance_ids = np.array(source_node_instance_ids[1:].split(','))
    sink_node_instance_ids = np.array(sink_node_instance_ids[:-1].split(','))
    instance_ids = np.array(instance_ids)
    source_node_ids = np.nonzero(source_node_instance_ids[:, None] == instance_ids)[1]
    sink_node_ids = np.nonzero(sink_node_instance_ids[:, None] == instance_ids)[1]

    edge_cid = relationship['relationship_cid']

    edge = np.array(np.meshgrid(source_node_ids, sink_node_ids)).T.reshape(-1, 2)
    edges.append(edge)
    edge_cids += [edge_cid]*edge.shape[0]
  edges = np.concatenate(edges, 0)

  edge_index = edges.T.astype(np.int64)
  edge_type = np.array(edge_cids, dtype=np.float32)

  y = np.array([subactivity_cid], dtype=np.int64)

  data = Data(x=torch.from_numpy(x),
              edge_index=torch.from_numpy(edge_index),
              edge_attr=torch.from_numpy(edge_type),
              y=torch.from_numpy(y))
  return data


class MomaTrimmedVideo(InMemoryDataset):
  def __init__(self, cfg):
    self.api = get_momaapi(cfg.data_dir, 'trimmed_video')
    cfg.num_actor_classes = len(self.api.actor_cnames)
    cfg.num_object_classes = len(self.api.object_cnames)
    cfg.num_relationship_classes = len(self.api.relationship_cnames)
    cfg.num_classes = len(self.api.subactivity_cnames)
    super(MomaTrimmedVideo, self).__init__(cfg.data_dir)


class MomaSpatialGraph(InMemoryDataset):
  def __init__(self, cfg, split):
    self.cfg = cfg
    self.split = split
    self.api = get_momaapi(cfg.data_dir, 'spatial_graph')
    cfg.num_actor_classes = len(self.api.actor_cnames)
    cfg.num_object_classes = len(self.api.object_cnames)
    cfg.num_relationship_classes = len(self.api.relationship_cnames)
    cfg.num_classes = len(self.api.subactivity_cnames)

    super(MomaSpatialGraph, self).__init__(cfg.data_dir)

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
                     self.cfg.num_actor_classes, self.cfg.num_object_classes, self.cfg.num_relationship_classes)
      data_list.append(data)

    data, slices = self.collate(data_list)
    torch.save((data, slices), self.processed_paths[0])
