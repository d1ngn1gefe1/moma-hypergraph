import numpy as np
import torch
from torchvision.io import read_video
from torch_geometric.data import InMemoryDataset, Data

from .moma_api import get_momaapi


def to_data(graph, num_actor_classes, num_object_classes, num_relationship_classes):
  node_cids = [actor['actor_cid'] for actor in graph['actors']]+ \
              [object['object_cid']+num_actor_classes for object in graph['objects']]
  instance_ids = [actor['instance_id'] for actor in graph['actors']]+ \
                 [object['instance_id'] for object in graph['objects']]

  num_node_features = num_actor_classes+num_object_classes
  num_edge_features = num_relationship_classes

  x = np.zeros((len(node_cids), num_node_features))
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
  edge_attr = np.zeros((len(edges), num_edge_features))
  edge_attr[np.arange(len(edge_cids)), edge_cids] = 1

  data = Data(torch.from_numpy(x), torch.from_numpy(edge_index), torch.from_numpy(edge_attr))
  return data


class MomaTrimmedVideo(InMemoryDataset):
  def __init__(self, dataset_dir):
    super(MomaTrimmedVideo, self).__init__(dataset_dir)
    self.api = get_momaapi(dataset_dir, 'trimmed_video')

  def len(self):
    return len(self.api.annotations)

  def get(self, index):
    pass





