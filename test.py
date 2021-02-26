from pprint import pprint
import random

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

  dataset = datasets.Moma(dataset_dir, level, True)
  video, annotation = dataset[0]
  print(video.shape, len(annotation))
  print(len(annotation['graphs']))


def main3():
  dataset_dir = '/home/ubuntu/datasets/moma'

  api = datasets.get_momaapi(dataset_dir, 'spatial_graph')

  ids = sorted(api.annotations.keys())
  id = random.choice(ids)

  annotation = api.get_annotation(id)
  datasets.Moma.parse_graph(annotation)
  print(annotation)


def main4():
  dataset_dir = '/home/ubuntu/datasets/MOMA'
  moma = datasets.MomaSpatialGraph(dataset_dir)


if __name__ == '__main__':
  main4()
