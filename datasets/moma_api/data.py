from dataclasses import dataclass
from itertools import chain
import numpy as np
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


@dataclass
class BBox:
  x1: int
  y1: int
  w: int
  h: int

  def __eq__(self, other):
    return self.__class__ == other.__class__ \
           and self.x1 == other.x1 and self.y1 == other.y1 \
           and self.w == other.w and self.h == other.h

  def __hash__(self):
    return hash((self.x1, self.y1, self.w, self.h))

  def resize(self, scale):
    return BBox(round(self.x1*scale), round(self.y1*scale), round(self.w*scale), round(self.h*scale))


@dataclass
class Entity:
  cid: int
  iid: str
  bbox: BBox

  def __post_init__(self):
    if not is_actor(self.iid) and not is_object(self.iid):
      raise ValueError

  def __eq__(self, other):
    return self.__class__ == other.__class__ \
           and self.cid == other.cid and self.iid == other.iid and self.bbox == other.bbox

  def __hash__(self):
    return hash((self.cid, self.iid, self.bbox))

  def resize(self, scale):
    return Entity(self.cid, self.iid, self.bbox.resize(scale))

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
    if any([not (is_actor(src_iid) or is_object(src_iid)) for src_iid in self.src_iids]) or \
       any([not (is_actor(snk_iid) or is_object(snk_iid)) for snk_iid in self.snk_iids]):
      raise ValueError

  def __eq__(self, other):
    return self.__class__ == other.__class__ \
           and self.cid == other.cid and self.src_iids == other.src_iids and self.snk_iids == other.snk_iids

  def __hash__(self):
    return hash((self.cid, *self.src_iids, *self.snk_iids))


@dataclass
class AG:
  actors: Set[Entity]
  objects: Set[Entity]
  relats: Set[Relat]

  def __post_init__(self):
    if not set(self.relat_entity_iids).issubset(set(self.actor_iids+self.object_iids)):
      raise ValueError

  @property
  def relat_entity_iids(self):
    return sorted(chain(*[relat.src_iids+relat.snk_iids for relat in self.relats]))

  @property
  def actor_iids(self):
    return sorted([actor.iid for actor in self.actors])

  @property
  def object_iids(self):
    return sorted([object.iid for object in self.objects])

  def resize(self, scale):
    return AG(set([actor.resize(scale) for actor in self.actors]),
              set([object.resize(scale) for object in self.objects]),
              self.relats)


@dataclass
class AAct:
  actor_iids: List[str]
  tracklets: np.ndarray  # [num_actors, num_frames]
  num_classes: int

  def __post_init__(self):
    if not all(is_actor(actor_iid) for actor_iid in self.actor_iids):
      raise ValueError

  def get_multilabels(self, per_frame) -> np.ndarray:
    multilabels = np.zeros((self.tracklets.shape[1], self.num_classes), dtype=np.int32)

    for j in range(self.tracklets.shape[1]):
      cids = sorted(set(chain(*[decode_aact(encoded_aact) for encoded_aact in self.tracklets[:, j]])))
      cids = [cid for cid in cids if cid >= 0]
      multilabels[j, cids] = 1

    # [num_frames, num_classes] -> [num_classes]
    if not per_frame:  # per video
      multilabels = np.sum(multilabels, axis=0)
      multilabels[multilabels > 0] = 1

    return multilabels
