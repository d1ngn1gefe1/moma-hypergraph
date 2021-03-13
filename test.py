from pprint import pprint
import random
from torchvision.io import read_video

import datasets


def main1():
  dataset_dir = '/home/ubuntu/datasets/MOMA/'

  api = datasets.get_momaapi(dataset_dir, 'untrimmed_video')
  print(api.annotations['0M1QrMIa7cg'])

  api = datasets.get_momaapi(dataset_dir, 'trimmed_video')
  print(api.annotations['20201115222849'])

  api = datasets.get_momaapi(dataset_dir, 'spatial_graph')
  print(api.annotations['20201117105640_1'])


def main2():
  dataset_dir = '/home/ubuntu/datasets/moma'
  level = 'trimmed_video'

  dataset = datasets.MOMA(dataset_dir, level, True)
  video, annotation = dataset[0]
  print(video.shape, len(annotation))
  print(len(annotation['graphs']))


def main3():
  dataset_dir = '/home/ubuntu/datasets/moma'

  api = datasets.get_momaapi(dataset_dir, 'spatial_graph')

  ids = sorted(api.annotations.keys())
  id = random.choice(ids)

  annotation = api.get_annotation(id)
  datasets.MOMA.parse_graph(annotation)
  print(annotation)


def main4():
  dataset_dir = '/home/ubuntu/datasets/MOMA'
  api = datasets.get_momaapi(dataset_dir, 'trimmed_video')

  lengths = []
  for trimmed_video_id, annotation in api.annotations.items():
    video_path = api.get_video_path(trimmed_video_id)
    video = read_video(video_path)[0]

    if video.shape[0] != len(annotation['graphs']):
      print('{}: {} vs. {}'.format(trimmed_video_id, video.shape[0], len(annotation['graphs'])))
      continue

    lengths.append(len(annotation['graphs']))

  lengths = sorted(lengths)
  print(lengths)


def main5():
  import numpy as np
  import torch
  from torch_geometric.data import DataLoader, Data, Batch

  num_nodes = 4
  num_features = 3
  num_edges = 5

  x = np.ones((num_nodes, num_features))
  edge_index = np.random.randint(0, num_nodes, (num_edges, num_features))
  data = Data(x=torch.from_numpy(x), edge_index=torch.from_numpy(edge_index))
  dataloader = DataLoader([data, data], batch_size=2)
  batch = next(iter(dataloader))
  print(edge_index)

  print(batch)
  print(batch.x)
  print(batch.edge_index)
  print(batch.batch)

  print('\n')

  batch = Batch.from_data_list([data, data])
  print(batch)
  print(batch.x)
  print(batch.edge_index)
  print(batch.batch)

  print('\n')

  print(batch.batch)
  setattr(data, 'batch_inner', data.batch)
  delattr(batch, 'batch')

  batch2d = Batch.from_data_list([batch, batch])
  print(batch2d)
  print(batch2d.x)
  print(batch2d.edge_index)
  print(batch2d.batch)

  print('\n')


def main6():
  import argparse
  import torch.utils.data
  import utils

  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', default='/home/ubuntu/datasets/MOMA', type=str)
  cfg = parser.parse_args()

  dataset = datasets.MOMATrim(cfg)

  # for video, trim_ann in dataset:
  #   assert video.shape[0] == len(trim_ann['ags'])

  dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, collate_fn=utils.collate_fn)
  videos, trim_anns = next(iter(dataloader))

  print(videos.shape)
  print([len(trim_ann['ags']) for trim_ann in trim_anns])


if __name__ == '__main__':
  main6()
