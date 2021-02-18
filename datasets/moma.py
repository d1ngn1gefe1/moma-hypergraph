from abc import ABC
import torch
from torchvision.io import read_video
# from torch_geometric.data import InMemoryDataset

from datasets.moma_api import get_momaapi


class Moma(ABC):
  def __init__(self):
    pass

  @staticmethod
  def parse_graph(annotation):
    # x, edge_index, edge_attr, y

    actor_instance_ids = []
    for actor in annotation['actors']:
      actor_instance_ids.append(actor['instance_id'])

    object_instance_ids = []
    for object in annotation['objects']:
      object_instance_ids.append(object['instance_id'])

    atomic_action_actor_instance_ids = []
    for action in annotation['atomic_actions']:
      atomic_action_actor_instance_ids.append(action['actor_instance_ids'])

    relationship_descriptions = []
    for relationship in annotation['relationships']:
      relationship_descriptions.append(relationship['description'])

    print(actor_instance_ids, object_instance_ids, atomic_action_actor_instance_ids, relationship_descriptions)


# class MomaSpatialGraph(InMemoryDataset, Moma):
#   def __init__(self, dataset_dir):
#     InMemoryDataset.__init__(self, dataset_dir)
#     Moma.__init__(self)
#     self.api = get_momaapi(dataset_dir, 'spatial_graph')
#
#   @property
#   def processed_file_names(self):
#     return ['data.pt']
#
#   def process(self):
#     # Read data into huge `Data` list.
#     data_list = [...]
#
#     ids = sorted(self.api.annotations.keys())
#
#     for id in ids:
#       annotation = self.api.get_annotation(id)
#
#     data, slices = self.collate(data_list)
#     torch.save((data, slices), self.processed_paths[0])

