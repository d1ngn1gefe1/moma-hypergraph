from abc import ABC
import json
import os

from .data_structure import *


class BaseAPI(ABC):
  """
  Acronym:
   - Data hierarchy
     - untrim: untrimmed-video-level
     - trim: trimmed-video-level
     - frame: frame-level
   - Task hierarchy
     - activity: act
     - sub-activity: sact
     - atomic action: aact
     - action hypergraph: ag
   - Graph
      - src: source node
      - snk: sink node
   - sample: sampled (video)
   - ann: annotations
   - iid: instance id
   - cid: class id
   - cname: class name
  """
  def __init__(self, data_dir, anns_dname='anns',
               untrim_dname='untrim_videos', trim_dname='trim_videos', trim_sample_dname='trim_sample_videos', feat_dname='feat'):
    # directories
    self.anns_dir = os.path.join(data_dir, anns_dname)
    self.untrim_dir = os.path.join(data_dir, untrim_dname)
    self.trim_dir = os.path.join(data_dir, trim_dname)
    self.trim_sample_dir = os.path.join(data_dir, trim_sample_dname)
    self.feat_dname = os.path.join(data_dir, feat_dname)

    # raw annotations
    self.raw_video_anns, self.raw_graph_anns = self.load_raw_anns()

    # indices
    self.act_cnames, self.sact_cnames,  self.aact_cnames, \
        self.actor_cnames, self.object_cnames, self.relat_cnames, \
        self.act_cids, self.sact_cids, self.untrim_ids, self.trim_ids = self.create_indices()

  def load_raw_anns(self, video_fname='video_anns.json', graph_fname='graph_anns.json'):
    """
    Load annotations
    """
    with open(os.path.join(self.anns_dir, video_fname), 'r') as f:
      raw_video_anns = json.load(f)

    with open(os.path.join(self.anns_dir, graph_fname), 'r') as f:
      raw_graph_anns = json.load(f)

    return raw_video_anns, raw_graph_anns

  @staticmethod
  def get_frame_id(raw_graph_ann):
    frame_id = '{}_{}'.format(raw_graph_ann['trim_video_id'], raw_graph_ann['frame_timestamp']-1)
    return frame_id

  def create_indices(self):
    """
    Arrays that map from class id (int) to class name (str)
      - act_cnames: act_cid -> act_cname
      - sact_cnames: sact_cid -> sact_cname
      - aact_cnames: aact_cid -> aact_cname
      - actor_cnames: actor_cid -> actor_cname
      - object_cnames: object_cid -> object_cname
      - relat_cnames: relat_cid -> relat_cname
    """
    act_cnames_path = os.path.join(self.anns_dir, 'act_cnames.txt')
    sact_cnames_path = os.path.join(self.anns_dir, 'sact_cnames.txt')
    aact_cnames_path = os.path.join(self.anns_dir, 'aact_cnames.txt')
    actor_cnames_path = os.path.join(self.anns_dir, 'actor_cnames.txt')
    object_cnames_path = os.path.join(self.anns_dir, 'object_cnames.txt')
    relat_cnames_path = os.path.join(self.anns_dir, 'relat_cnames.txt')

    """
    Dicts that map from id (str) to class id (int)
      - act_cids: untrim_id -> act_cid
      - sact_cids: trim_id -> sact_cid
    """
    act_cids_path = os.path.join(self.anns_dir, 'act_cids.json')
    sact_cids_path = os.path.join(self.anns_dir, 'sact_cids.json')

    """
    Dicts that map from lower-level id (str) to higher-level id (str)
      - untrim_ids: trim_id -> untrim_id
      - trim_ids: frame_id -> trim_id
    """
    untrim_ids_path = os.path.join(self.anns_dir, 'untrim_ids.json')
    trim_ids_path = os.path.join(self.anns_dir, 'trim_ids.json')

    # load pre-extracted indices
    if all(list(map(os.path.isfile, [act_cnames_path, sact_cnames_path, aact_cnames_path,
                                     actor_cnames_path, object_cnames_path, relat_cnames_path,
                                     act_cids_path, sact_cids_path,
                                     untrim_ids_path, trim_ids_path]))):
      print('Load indices')
      
      with open(act_cnames_path, 'r') as f_act_cnames, \
           open(sact_cnames_path, 'r') as f_sact_cnames, \
           open(aact_cnames_path, 'r') as f_aact_cnames, \
           open(actor_cnames_path, 'r') as f_actor_cnames, \
           open(object_cnames_path, 'r') as f_object_cnames, \
           open(relat_cnames_path, 'r') as f_relat_cnames:
        act_cnames = f_act_cnames.read().splitlines()
        sact_cnames = f_sact_cnames.read().splitlines()
        aact_cnames = f_aact_cnames.read().splitlines()
        actor_cnames = f_actor_cnames.read().splitlines()
        object_cnames = f_object_cnames.read().splitlines()
        relat_cnames = f_relat_cnames.read().splitlines()

      with open(act_cids_path, 'r') as f_act_cids, \
           open(sact_cids_path, 'r') as f_sact_cids:
        act_cids = json.load(f_act_cids)
        sact_cids = json.load(f_sact_cids)

      with open(untrim_ids_path, 'r') as f_untrim_ids, \
           open(trim_ids_path, 'r') as f_trim_ids:
        untrim_ids = json.load(f_untrim_ids)
        trim_ids = json.load(f_trim_ids)

      return act_cnames, sact_cnames, aact_cnames, \
             actor_cnames, object_cnames, relat_cnames, \
             act_cids, sact_cids, untrim_ids, trim_ids

    # extract indices
    act_cnames, sact_cnames = [], []
    for untrim_id, raw_video_ann in self.raw_video_anns.items():
      act_cnames.append(raw_video_ann['class'])
      for sact in raw_video_ann['subactivity']:
        sact_cnames.append(sact['class'])

    actor_cnames, object_cnames, aact_cnames, relat_cnames = [], [], [], []
    for raw_graph_ann in self.raw_graph_anns:
      actor_cnames += [actor['class'] for actor in raw_graph_ann['annotation']['actors']]
      object_cnames += [object['class'] for object in raw_graph_ann['annotation']['objects']]
      aact_cnames += [aact['class'] for aact in raw_graph_ann['annotation']['atomic_actions']]
      relat_cnames += [relat['class'] for relat in raw_graph_ann['annotation']['relationships']]

    act_cnames = sorted(list(set(act_cnames)))
    sact_cnames = sorted(list(set(sact_cnames)))
    actor_cnames = sorted(list(set(actor_cnames)))
    object_cnames = sorted(list(set(object_cnames)))
    aact_cnames = sorted(list(set(aact_cnames)))
    relat_cnames = sorted(list(set(relat_cnames)))

    act_cids, sact_cids, untrim_ids, trim_ids = {}, {}, {}, {}

    for untrim_id, raw_video_ann in self.raw_video_anns.items():
      if untrim_id not in act_cids:
        act_cids[untrim_id] = act_cnames.index(raw_video_ann['class'])

      for sact in raw_video_ann['subactivity']:
        trim_id = sact['trim_video_id']

        if trim_id not in sact_cids:
          sact_cids[trim_id] = sact_cnames.index(sact['class'])

        if trim_id not in untrim_ids:
          untrim_ids[trim_id] = untrim_id

    for raw_graph_ann in self.raw_graph_anns:
      trim_id = raw_graph_ann['trim_video_id']
      frame_id = self.get_frame_id(raw_graph_ann)

      if frame_id not in trim_ids:
        trim_ids[frame_id] = trim_id

    print('Generate indices')
    with open(act_cnames_path, 'w') as f_act_cnames, \
         open(sact_cnames_path, 'w') as f_sact_cnames, \
         open(aact_cnames_path, 'w') as f_aact_cnames, \
         open(actor_cnames_path, 'w') as f_actor_cnames, \
         open(object_cnames_path, 'w') as f_object_cnames, \
         open(relat_cnames_path, 'w') as f_relat_cnames:
      f_act_cnames.write('\n'.join(act_cnames))
      f_sact_cnames.write('\n'.join(sact_cnames))
      f_aact_cnames.write('\n'.join(aact_cnames))
      f_actor_cnames.write('\n'.join(actor_cnames))
      f_object_cnames.write('\n'.join(object_cnames))
      f_relat_cnames.write('\n'.join(relat_cnames))

    with open(act_cids_path, 'w') as f_act_cids, \
         open(sact_cids_path, 'w') as f_sact_cids:
      f_act_cids.write(json.dumps(act_cids))
      f_sact_cids.write(json.dumps(sact_cids))

    with open(untrim_ids_path, 'w') as f_untrim_ids, \
         open(trim_ids_path, 'w') as f_trim_ids:
      f_untrim_ids.write(json.dumps(untrim_ids))
      f_trim_ids.write(json.dumps(trim_ids))

    return act_cnames, sact_cnames, aact_cnames, \
           actor_cnames, object_cnames,  relat_cnames, \
           act_cids, sact_cids, untrim_ids, trim_ids

  @staticmethod
  def parse_bbox(bbox: Dict, size: Size) -> BBox:
    x = [bbox['topLeft']['x'], bbox['bottomLeft']['x'], bbox['topRight']['x'], bbox['bottomRight']['x']]
    y = [bbox['topLeft']['y'], bbox['bottomLeft']['y'], bbox['topRight']['y'], bbox['bottomRight']['y']]

    x1 = max(round(min(x)), 0)
    y1 = max(round(min(y)), 0)
    w = min(round(max(x)-x1+1), size.w-1)
    h = min(round(max(y)-y1+1), size.h-1)

    return BBox(x1, y1, w, h)

  @staticmethod
  def parse_size(size: dict) -> Size:
    return Size(size['width'], size['height'])

  def parse_actor(self, actor: Dict, size: Size) -> Entity:
    cid = self.actor_cnames.index(actor['class'])
    iid = actor['id_in_video']
    bbox = self.parse_bbox(actor['bbox'], size)

    return Entity(cid, iid, bbox)

  def parse_object(self, object: Dict, size: Size) -> Entity:
    cid = self.object_cnames.index(object['class'])
    iid = object['id_in_video']
    bbox = self.parse_bbox(object['bbox'], size)

    return Entity(cid, iid, bbox)

  def parse_relat(self, relat: Dict) -> Relationship:
    cname = relat['class']
    description = relat['description']

    cid = self.relat_cnames.index(cname)
    src_iids, snk_iids = description[1:-1].split('),(')
    src_iids = set(src_iids.split(','))
    snk_iids = set(snk_iids.split(','))

    return Relationship(cid, src_iids, snk_iids)

  def parse_aact(self, aact: Dict) -> AAct:
    cid = self.aact_cnames.index(aact['class']),
    actor_iids = set(aact['actor_id'].split(','))

    return AAct(cid, actor_iids)

  def parse_ag(self, raw_graph_ann):
    size = self.parse_size(raw_graph_ann['frame_dim'])
    aacts = [self.parse_aact(aact) for aact in raw_graph_ann['annotation']['atomic_actions']]
    actors = [self.parse_actor(actor, size) for actor in raw_graph_ann['annotation']['actors']]
    objects = [self.parse_object(object, size) for object in raw_graph_ann['annotation']['objects']]
    relats = [self.parse_relat(relat) for relat in raw_graph_ann['annotation']['relationships']]
    ag = AG(aacts, actors, objects, relats)

    # sanity check
    actor_iids = set([actor['id_in_video'] for actor in raw_graph_ann['annotation']['actors']])
    object_iids = set([object['id_in_video'] for object in raw_graph_ann['annotation']['objects']])
    entity_iids = actor_iids.union(object_iids)
    assert all([aact.actor_iids.issubset(actor_iids) for aact in ag.aacts]), [aact.actor_iids.issubset(actor_iids) for aact in ag.aacts]
    assert all([relat.src_iids.issubset(entity_iids) for relat in relats])
    assert all([relat.snk_iids.issubset(entity_iids) for relat in relats])

    return ag
