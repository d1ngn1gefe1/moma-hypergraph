from collections import namedtuple
from dataclasses import dataclass
from typing import Dict, List, Set


BBox = namedtuple('BBox', ['x1', 'y1', 'w', 'h'])
Size = namedtuple('Size', ['w', 'h'])


@dataclass
class Actor:
  cid: int
  iid: int
  bbox: BBox


@dataclass
class Object:
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
  actors: List[Actor]
  objects: List[Object]
  relationships: List[Relationship]
