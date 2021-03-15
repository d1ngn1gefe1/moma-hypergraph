import numpy as np
import torch
from torch.utils.data.dataloader import default_collate
from torch._six import container_abcs, string_classes, int_classes
from torch_geometric.data import Batch, Data



def cat_vl(tensor_list):
  """ Concatenate tensors of varying lengths
  :param tensor_list: a list of tensors of varying tensor.shape[0] but same tensor.shape[1:]
  :return: a concatenated tensor and chunk sizes
  """
  chunk_sizes = torch.IntTensor([tensor.shape[0] for tensor in tensor_list])
  tensor = torch.cat(tensor_list, dim=0)
  return tensor, chunk_sizes


def split_vl(tensor, chunk_sizes):
  """ Split a tensor into sub-tensors of varying lengths
  """
  if isinstance(chunk_sizes, torch.Tensor) or isinstance(chunk_sizes, np.ndarray):
    chunk_sizes = chunk_sizes.tolist()
  return list(torch.split(tensor, chunk_sizes))


def collate_fn(batch):
  elem = batch[0]

  if isinstance(elem, Data):
    return Batch.from_data_list(batch)
  elif isinstance(elem, torch.Tensor):
    return default_collate(batch)
  elif isinstance(elem, float):
    return torch.tensor(batch, dtype=torch.float)
  elif isinstance(elem, int_classes):
    return torch.tensor(batch)
  elif isinstance(elem, string_classes):
    return batch
  elif isinstance(elem, container_abcs.Mapping):
    # return {key: collate_fn([d[key] for d in batch]) for key in elem}
    return batch
  elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
    print('b')
    return type(elem)(*(collate_fn(s) for s in zip(*batch)))
  elif isinstance(elem, container_abcs.Sequence):
    print('c')
    return [collate_fn(s) for s in zip(*batch)]

  raise TypeError('DataLoader found invalid type: {}'.format(type(elem)))


def to_pyg_data(trim_ann, feat, y):
  """
   - edge_index: [2, num_edges]
   - node_feature: [num_nodes, num_node_features]
   - edge_feature: [num_edges, num_edge_features]
   - node_label: [num_nodes, num_node_classes]
   - edge_label: [num_edges]
   -
  """
  chunk_sizes = [ag.num_nodes for ag in trim_ann['ags']]
  feat_list = split_vl(feat, chunk_sizes)
  data_list = []

  for ag, feat in zip(trim_ann['ags'], feat_list):
    node_feature = feat
    node_label = ag.entity_cids
    edge_index, edge_label = ag.pairwise_edges

    # print(edge_index.shape, node_feature.shape, node_label.shape, edge_label.shape)
    data = Data(x=node_feature, edge_index=edge_index)
    data_list.append(data)

  data = Batch.from_data_list(data_list)
  setattr(data, 'y', y)
  setattr(data, 'batch_temporal', data.batch)
  delattr(data, 'batch')

  return data


# def to_pyg_data(ag, subactivity_cid, num_actor_classes, num_object_classes, num_relat_classes):
#   actor_cids =
#   entity_cids = [actor['cid'] for actor in ag['actors']]+ \
#               [object['cid']+num_actor_classes for object in ag['objects']]
#   instance_ids = [actor['iid'] for actor in ag['actors']]+ \
#                  [object['iid'] for object in ag['objects']]
#
#   num_node_features = num_actor_classes+num_object_classes
#   num_edge_features = num_relat_classes
#
#   x = np.zeros((len(entity_cids), num_node_features), dtype=np.float32)
#   x[np.arange(len(entity_cids)), entity_cids] = 1
#
#   edges, edge_cids = [], []
#   for relat in graph['relationships']:
#     source_node_instance_ids, sink_node_instance_ids = relat['description'].split('),(')
#     source_node_instance_ids = np.array(source_node_instance_ids[1:].split(','))
#     sink_node_instance_ids = np.array(sink_node_instance_ids[:-1].split(','))
#     instance_ids = np.array(instance_ids)
#     source_node_ids = np.nonzero(source_node_instance_ids[:, None] == instance_ids)[1]
#     sink_node_ids = np.nonzero(sink_node_instance_ids[:, None] == instance_ids)[1]
#
#     edge_cid = relat['relat_cid']
#
#     edge = np.array(np.meshgrid(source_node_ids, sink_node_ids)).T.reshape(-1, 2)
#     edges.append(edge)
#     edge_cids += [edge_cid]*edge.shape[0]
#   edges = np.concatenate(edges, 0)
#
#   edge_index = edges.T.astype(np.int64)
#   edge_type = np.array(edge_cids, dtype=np.float32)
#
#   y = np.array([subactivity_cid], dtype=np.int64)
#
#   data = Data(x=torch.from_numpy(x),
#               edge_index=torch.from_numpy(edge_index),
#               edge_attr=torch.from_numpy(edge_type),
#               y=torch.from_numpy(y))
#   return data