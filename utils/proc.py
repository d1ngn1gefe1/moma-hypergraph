import numpy as np
import torch
from torch_geometric.data import Data


def to_geometric(ahg, subactivity_cid, num_actor_classes, num_object_classes, num_relationship_classes):
  actor_cids =
  entity_cids = [actor['cid'] for actor in ahg['actors']]+ \
              [object['cid']+num_actor_classes for object in ahg['objects']]
  instance_ids = [actor['iid'] for actor in ahg['actors']]+ \
                 [object['iid'] for object in ahg['objects']]

  num_node_features = num_actor_classes+num_object_classes
  num_edge_features = num_relationship_classes

  x = np.zeros((len(entity_cids), num_node_features), dtype=np.float32)
  x[np.arange(len(entity_cids)), entity_cids] = 1

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