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
