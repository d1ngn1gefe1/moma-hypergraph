from itertools import chain
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
  chunk_sizes = torch.LongTensor([tensor.shape[0] for tensor in tensor_list])
  tensor = torch.cat(tensor_list, dim=0)
  return tensor, chunk_sizes


def split_vl(tensor, chunk_sizes):
  """ Split a tensor into sub-tensors of varying lengths
  """
  if isinstance(chunk_sizes, torch.Tensor) or isinstance(chunk_sizes, np.ndarray):
    chunk_sizes = chunk_sizes.tolist()
  return list(torch.split(tensor, chunk_sizes))


def to_batch(chunk_sizes):
  batch = list(chain(*[[i]*chunk_size for i, chunk_size in enumerate(chunk_sizes)]))
  batch = torch.LongTensor(batch)
  return batch.to(chunk_sizes.device)


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


def to_pyg_data(trim_ann, feat, act_cid, sact_cid,
                num_actor_classes, num_object_classes, num_relat_classes):
  """
   - edge_index: [2, num_edges]
   - node_attr: [num_nodes, num_node_attrs]
   - edge_attr: [num_edges, num_edge_attrs]
   - node_label: [num_nodes]
   - edge_label: [num_edges]
   - orc_node_attr: [num_nodes, num_actor_classes+num_object_classes], one-hot
   - orc_edge_attr: [num_edges, num_relat_classes], one-hot
  """

  node_frame_chunk_sizes = [ag.num_nodes for ag in trim_ann['ags']]  # chunk sizes that split nodes by frame
  actor_iids = trim_ann['aact'].actor_iids  # list
  actor_cids = trim_ann['aact'].actor_cids  # list
  feat_list = split_vl(feat, node_frame_chunk_sizes)
  data_list = []
  batch_actor = []
  hyperedge_chunk_sizes = []

  # create a list of Data objects, one for each graph (frame)
  for ag, feat in zip(trim_ann['ags'], feat_list):
    node_attr = feat
    node_label = torch.LongTensor(ag.actor_cids+[object_cid+num_actor_classes for object_cid in ag.object_cids])
    edge_index, edge_label, hyperedge_label, edge_attr, chunk_sizes = ag.edges
    hyperedge_chunk_sizes += chunk_sizes  # split edges by hyperedge

    orc_node_attr = torch.zeros(feat.shape[0], num_actor_classes+num_object_classes)
    orc_node_attr[torch.arange(feat.shape[0]), node_label] = 1

    orc_edge_attr = torch.zeros(edge_index.shape[1], num_relat_classes)
    orc_edge_attr[torch.arange(edge_index.shape[1]), torch.LongTensor(edge_label)] = 1

    # the first index is for all objects
    actor_indices = [0]*len(ag.object_iids)+[actor_iids.index(actor_iid)+1 for actor_iid in ag.actor_iids]
    batch_actor.append(actor_indices)

    data = Data(edge_index=edge_index,
                x=node_attr, orc_node_attr=orc_node_attr, node_label=node_label,
                edge_attr=edge_attr, orc_edge_attr=orc_edge_attr, edge_label=edge_label,
                hyperedge_label=hyperedge_label)
    data_list.append(data)

  # concatenate graphs from a video into a single graph
  data = Batch.from_data_list(data_list)

  # calculate useful attributes
  node_video_chunk_sizes = sum(node_frame_chunk_sizes)
  hyperedge_chunk_sizes = torch.cat(hyperedge_chunk_sizes, dim=0) if len(hyperedge_chunk_sizes) > 0 else torch.LongTensor(0)
  ps_aact_cids = torch.from_numpy(trim_ann['aact'].get_ps_labels(frame_level=False))
  pa_aact_cids = torch.from_numpy(trim_ann['aact'].get_pa_labels(frame_level=False))
  actor_cids = torch.LongTensor(actor_cids)
  batch_frame = data.batch  # length=num_nodes, scatter by frame id
  batch_actor = torch.LongTensor(list(chain.from_iterable(batch_actor)))  # length=num_nodes, scatter by actor iid

  # cheat the assert statement so that batches can be further batched
  delattr(data, 'batch')

  # set useful attributes
  setattr(data, 'node_video_chunk_sizes', node_video_chunk_sizes)  # split nodes by video
  setattr(data, 'hyperedge_chunk_sizes', hyperedge_chunk_sizes)  # split edges by hyperedge
  setattr(data, 'act_cids', act_cid)
  setattr(data, 'sact_cids', sact_cid)
  setattr(data, 'ps_aact_cids', ps_aact_cids)
  setattr(data, 'pa_aact_cids', pa_aact_cids)
  setattr(data, 'actor_cids', actor_cids)
  setattr(data, 'batch_frame', batch_frame)  # scatter per-video feat by frame id
  setattr(data, 'batch_actor', batch_actor)  # scatter per-video feat by actor iid

  return data
