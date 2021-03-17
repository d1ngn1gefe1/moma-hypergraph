from dataclasses import dataclass
from itertools import chain, product
import numpy as np
import torch
from typing import Dict, List, Set, Tuple


""" Entity types
"""


def is_actor(iid):
  return iid.islower() and len(iid) == 1


def is_object(iid):
  return iid.isnumeric()


""" Atomic action encoding
 - actor absent: [-2]
 - actor present + inactive: [-1]
 - actor present + active: a sorted list of aact_cids
"""


def decode_aact(encoded_aact: np.int64, n: int=2) -> List[int]:
  assert encoded_aact >= -2
  if encoded_aact < 0:
    return [int(encoded_aact)]
  else:
    decoded_aact = sorted(set([int(str(encoded_aact)[max(i-n, 0):i])
                               for i in reversed(range(len(str(encoded_aact)), 0, -n))]))
    return decoded_aact


def encode_aact(decoded_aact: List[int], n: int=2) -> np.int64:
  if decoded_aact[0] < 0:
    assert len(decoded_aact) == 1 and decoded_aact[0] >= -2
    return np.int64(decoded_aact)
  else:
    assert all([x >= 0 and len(str(x)) <= n for x in decoded_aact])
    encoded_aact = np.int64(''.join([str(x).zfill(n) for x in sorted(set(decoded_aact))]))
    return encoded_aact


def add_encoded_aact(encoded_aact: np.int64, aact_cid: int):
  assert encoded_aact >= -1, encoded_aact  # present
  if encoded_aact == -1:  # first atomic action
    return aact_cid
  else:  # more than 1 atomic action
    decoded_aact = sorted(set(decode_aact(encoded_aact)+[aact_cid]))
    encoded_aact = encode_aact(decoded_aact)
    return encoded_aact


""" Data structures
"""


@dataclass
class Size:
  w: int
  h: int

  def resize(self, scale: float):
    return Size(round(self.w*scale), round(self.h*scale))


@dataclass
class BBox:
  x1: int
  y1: int
  w: int
  h: int

  def __post_init__(self):
    if self.w <= 0 or self.h <= 0:
      raise ValueError

  def __eq__(self, other):
    return self.__class__ == other.__class__ \
           and self.x1 == other.x1 and self.y1 == other.y1 \
           and self.w == other.w and self.h == other.h

  def __hash__(self):
    return hash((self.x1, self.y1, self.w, self.h))

  def resize(self, scale: float):
    return BBox(round(self.x1*scale), round(self.y1*scale), round(self.w*scale), round(self.h*scale))

  def distance(self, other):
    dx = (self.x1+self.w//2)-(other.x1+other.w//2)
    dy = (self.y1+self.h//2)-(other.y1+other.h//2)
    return np.sqrt(dx**2+dy**2)


@dataclass
class Entity:
  cid: int
  iid: str
  bbox: BBox

  def __post_init__(self):
    # wrong iid format
    if not is_actor(self.iid) and not is_object(self.iid):
      raise ValueError

  def __lt__(self, other):
    return self.iid < other.iid

  def __eq__(self, other):
    return self.__class__ == other.__class__ and self.iid == other.iid

  def __hash__(self):
    return hash(self.iid)

  def resize(self, scale: float):
    return Entity(self.cid, self.iid, self.bbox.resize(scale))

  def __repr__(self):
    return f'{self.type}(cid={self.cid}, iid={self.iid})'

  @property
  def type(self):
    if is_object(self.iid):
      return 'Object'
    else:  # is_actor(self.iid)
      return 'Actor'


@dataclass
class Relat:
  cid: int
  src_iids: List[str]
  snk_iids: List[str]

  def __post_init__(self):
    # wrong iid format
    if any([not (is_actor(src_iid) or is_object(src_iid)) for src_iid in self.src_iids]) or \
       any([not (is_actor(snk_iid) or is_object(snk_iid)) for snk_iid in self.snk_iids]):
      print('src_iids={}, snk_iids={}'.format(self.src_iids, self.snk_iids))
      raise ValueError

    # not in order or duplicate
    if self.src_iids != sorted(set(self.src_iids)) or self.snk_iids != sorted(set(self.snk_iids)):
      print('src_iids={}, snk_iids={}'.format(self.src_iids, self.snk_iids))
      raise ValueError

  def __repr__(self):
    return f'Relat(cid={self.cid}), {self.src_iids}->{self.snk_iids}'

  def __eq__(self, other):
    return self.__class__ == other.__class__ \
           and self.cid == other.cid and self.src_iids == other.src_iids and self.snk_iids == other.snk_iids

  def __hash__(self):
    return hash((self.cid, *self.src_iids, *self.snk_iids))


@dataclass
class AG:
  actors: List[Entity]
  objects: List[Entity]
  relats: Set[Relat]

  def __post_init__(self):
    if not set(self.relat_entity_iids).issubset(set(self.actor_iids+self.object_iids)):
      raise ValueError

    # not in order or duplicate
    if self.actors != sorted(set(self.actors)) or self.objects != sorted(set(self.objects)):
      print('actors={}, objects={}'.format(self.actors, self.objects))
      raise ValueError

  def __repr__(self):
    message = (
        f'AG(\n'
        f'\tactors={self.actors}\n'
        f'\tobjects={self.objects}\n'
        f'\trelats={list(self.relats)}\n'
        f')'
    )
    return message

  @property
  def entities(self):
    return self.actors+self.objects

  @property
  def relat_entity_iids(self):
    return sorted(chain(*[relat.src_iids+relat.snk_iids for relat in self.relats]))

  @property
  def actor_iids(self):
    return [actor.iid for actor in self.actors]  # already sorted

  @property
  def object_iids(self):
    return [object.iid for object in self.objects]  # already sorted

  @property
  def entity_iids(self):
    return self.actor_iids+self.object_iids  # same order as attr

  @property
  def actor_cids(self):
    return [actor.cid for actor in self.actors]

  @property
  def object_cids(self):
    return [object.cid for object in self.objects]

  @property
  def num_nodes(self):
    return len(self.entities)

  @property
  def num_edges(self):
    return len(self.relats)

  @property
  def edges(self):
    edge_index = []
    edge_label = []
    hyperedge_label = []
    edge_attr = []
    hyperedge_chunk_sizes = []

    for relat in self.relats:
      src_indices = [self.entity_iids.index(iid) for iid in relat.src_iids]
      snk_indices = [self.entity_iids.index(iid) for iid in relat.snk_iids]
      edge_index.append(torch.LongTensor(list(product(src_indices, snk_indices))).T)
      edge_label += [relat.cid]*len(src_indices)*len(snk_indices)
      hyperedge_label.append(relat.cid)

      src_bboxes = [self.entities[src_index].bbox for src_index in src_indices]
      snk_bboxes = [self.entities[snk_index].bbox for snk_index in snk_indices]

      for src_bbox in src_bboxes:
        for snk_bbox in snk_bboxes:
          distance = src_bbox.distance(snk_bbox)
          edge_attr.append([src_bbox.x1, src_bbox.y1, src_bbox.w, src_bbox.h,
                            snk_bbox.x1, snk_bbox.y1, snk_bbox.w, snk_bbox.h,
                            distance])

      hyperedge_chunk_sizes.append([len(src_indices)*len(snk_indices)])

    if len(edge_index) == 0:
      edge_index = torch.LongTensor(2, 0)
      edge_label = torch.LongTensor(0)
      hyperedge_label = torch.LongTensor(0)
      edge_attr = torch.FloatTensor(0, 9)
      hyperedge_chunk_sizes = torch.LongTensor(0)
    else:
      edge_index = torch.cat(edge_index, dim=1)
      edge_label = torch.LongTensor(edge_label)
      hyperedge_label = torch.LongTensor(hyperedge_label)
      edge_attr = torch.FloatTensor(edge_attr)
      hyperedge_chunk_sizes = torch.LongTensor(hyperedge_chunk_sizes)

    return edge_index, edge_label, hyperedge_label, edge_attr, hyperedge_chunk_sizes

  def resize(self, scale: float):
    return AG([actor.resize(scale) for actor in self.actors],
              [object.resize(scale) for object in self.objects],
              self.relats)


@dataclass
class AAct:
  actor_iids: List[str]
  actor_cids: List[str]
  encoded_tracklets: np.ndarray  # [num_actors, num_frames]
  num_classes: int

  def __post_init__(self):
    if not all(is_actor(actor_iid) for actor_iid in self.actor_iids):
      raise ValueError
    if self.actor_iids != sorted(set(self.actor_iids)):
      raise ValueError

  def __repr__(self):
    message = (
      f'AAct(\n'
      f'\tactor_iids={self.actor_iids}\n'
      f'\tactor_cids={self.actor_cids}\n'
      f'\tnum_actors={self.encoded_tracklets.shape[0]}\n'
      f'\tnum_frames={self.encoded_tracklets.shape[1]}\n'
      f')'
    )
    return message

  def get_ps_labels(self, frame_level: bool=True) -> np.ndarray:
    """ Per-scene multi-labels
    Return:
     - frame-level: shape=[num_frames, num_classes], range=[0, 1]
     - video-level: shape=[1, num_classes], range=[0, 1]
    """
    num_frames = self.encoded_tracklets.shape[1]
    labels = np.zeros((num_frames, self.num_classes))

    for j in range(num_frames):
      cids = sorted(set(chain(*[decode_aact(encoded_aact) for encoded_aact in self.encoded_tracklets[:, j]])))
      cids = [cid for cid in cids if cid >= 0]
      labels[j, cids] = 1

    # [num_frames, num_classes] -> [num_classes]
    if not frame_level:  # per video
      labels = np.sum(labels, axis=-2, keepdims=True)
      labels[labels > 0] = 1

    return labels

  def get_pa_labels(self, frame_level: bool=True) -> np.ndarray:
    """ Per-actor multi-labels
    Return:
     - frame-level: shape=[num_actors, num_frames, num_classes], range=[0, 1]
     - video-level: shape=[num_actors, num_classes], range=[0, 1]
    """
    num_actors, num_frames = self.encoded_tracklets.shape
    labels = np.zeros((num_actors, num_frames, self.num_classes))

    for i in range(num_actors):
      for j in range(num_frames):
        cids = decode_aact(self.encoded_tracklets[i, j])
        cids = [cid for cid in cids if cid >= 0]
        labels[i, j, cids] = 1

    # [num_actors, num_frames, num_classes] -> [num_actors, num_classes]
    if not frame_level:  # per video
      labels = np.sum(labels, axis=-2)
      labels[labels > 0] = 1

    return labels
