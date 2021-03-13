from dataclasses import dataclass
from typing import Dict, List, Set


@dataclass
class BBox:
  x1: int
  y1: int
  w: int
  h: int

  def resize(self, scale):
    return BBox(round(self.x1*scale), round(self.y1*scale), round(self.w*scale), round(self.h*scale))


@dataclass
class Size:
  w: int
  h: int


@dataclass
class Entity:
  cid: int
  iid: int
  bbox: BBox

  def resize(self, scale):
    return Entity(self.cid, self.iid, self.bbox.resize(scale))


@dataclass
class Relationship:
  cid: int
  src_iids: Set
  snk_iids: Set


@dataclass
class AAct:
  cid: int
  actor_iids: Set


@dataclass
class AG:
  aacts: List[AAct]
  actors: List[Entity]
  objects: List[Entity]
  relats: List[Relationship]

  def resize(self, scale):
    return AG(self.aacts,
              [actor.resize(scale) for actor in self.actors],
              [object.resize(scale) for object in self.objects],
              self.relats)
