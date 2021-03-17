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
  edge_index = np.random.randint(0, num_nodes, (2, num_features))
  data1 = Data(x=torch.from_numpy(x), edge_index=torch.from_numpy(edge_index))
  data2 = Data(x=torch.from_numpy(x), edge_index=torch.LongTensor(2, 0))
  data3 = data1
  dataloader = DataLoader([data1, data2, data3], batch_size=3)
  batch = next(iter(dataloader))
  print(edge_index)

  print(batch)
  print(batch.x)
  print(batch.edge_index)
  print(batch.batch)

  # print('\n')
  #
  # batch = Batch.from_data_list([data, data])
  # print(batch)
  # print(batch.x)
  # print(batch.edge_index)
  # print(batch.batch)
  #
  # print('\n')
  #
  # print(batch.batch)
  # setattr(data, 'batch_inner', data.batch)
  # delattr(batch, 'batch')
  #
  # batch2d = Batch.from_data_list([batch, batch])
  # print(batch2d)
  # print(batch2d.x)
  # print(batch2d.edge_index)
  # print(batch2d.batch)
  #
  # print('\n')


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


def main7():
  import torch
  from utils import PyGData
  from torch_geometric.data import Batch

  x1 = torch.zeros(5, 10)  # [num_nodes, num_features]
  x2 = torch.zeros(3, 10)  # [num_nodes, num_features]
  edge_index1 = torch.LongTensor([[0, 1, 0, 1, 0, 1],
                                  [0, 0, 1, 1, 2, 2]])
  edge_index2 = torch.LongTensor([[0, 1, 0, 1, 3, 4],
                                  [0, 0, 1, 1, 2, 2]])

  data1 = PyGData(x=x1, edge_index=edge_index1)
  data2 = PyGData(x=x2, edge_index=edge_index2)
  data_list = [data1, data2]
  data = Batch.from_data_list(data_list)

  print(data.edge_index)


if __name__ == '__main__':
  main7()
