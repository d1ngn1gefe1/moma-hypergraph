from dataclasses import dataclass
from typing import Dict, List, Set


@dataclass
class BBox:
  x1: int
  y1: int
  w: int
  h: int


@dataclass
class Size:
  w: int
  h: int


@dataclass
class Entity:
  cid: int
  iid: int
  bbox: BBox


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
class AHG:
  aacts: List[AAct]
  actors: List[Entity]
  objects: List[Entity]
  relationships: List[Relationship]
