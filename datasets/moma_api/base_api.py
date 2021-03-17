from abc import ABC
import json
import os

from .data import *


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
       - src: source node
       - snk: sink node
       - relat: relationship
   - sample: sampled (video)
   - ann: annotations
   - iid: instance id
   - cid: class id
   - cname: class name
  """
  def __init__(self, data_dir, split_by, feats_dname='feats', anns_dname='anns', untrim_dname='untrim_videos',
               trim_dname='trim_videos', trim_sample_dname='trim_sample_videos'):
    # directories
    self.anns_dir = os.path.join(data_dir, anns_dname)
    self.untrim_dir = os.path.join(data_dir, untrim_dname)
    self.trim_dir = os.path.join(data_dir, trim_dname)
    self.trim_sample_dir = os.path.join(data_dir, trim_sample_dname)
    self.feats_dir = os.path.join(data_dir, feats_dname)

    # raw annotations
    self.raw_video_anns, self.raw_graph_anns = self.load_raw_anns()

    # indices
    self.act_cnames, self.sact_cnames,  self.aact_cnames, \
        self.actor_cnames, self.object_cnames, self.relat_cnames, \
        self.act_cids, self.sact_cids, self.untrim_ids, self.trim_ids = self.create_indices()

    # splits
    self.split_train, self.split_val = self.load_splits(split_by)

  def load_raw_anns(self, video_fname='video_anns.json', graph_fname='graph_anns.json'):
    """
    Load annotations
    """
    with open(os.path.join(self.anns_dir, video_fname), 'r') as f:
      raw_video_anns = json.load(f)

    with open(os.path.join(self.anns_dir, graph_fname), 'r') as f:
      raw_graph_anns = json.load(f)

    return raw_video_anns, raw_graph_anns

  def load_splits(self, split_by, train_fname='train.txt', val_fname='val.txt'):
    with open(os.path.join(self.anns_dir, 'split_by_{}'.format(split_by), train_fname), 'r') as f_train, \
         open(os.path.join(self.anns_dir, 'split_by_{}'.format(split_by), val_fname), 'r') as f_val:
      split_train = f_train.read().splitlines()
      split_val = f_val.read().splitlines()

    assert sorted(split_train+split_val) == sorted(self.sact_cids.keys())
    print('{}: len(train) = {}, len(val) = {}'.format('Split by {}'.format(split_by), len(split_train), len(split_val)))

    return split_train, split_val

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
      print('Load MOMA API indices')
      
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

      with open(act_cids_path, 'r') as f_act_cids, open(sact_cids_path, 'r') as f_sact_cids:
        act_cids = json.load(f_act_cids)
        sact_cids = json.load(f_sact_cids)

      with open(untrim_ids_path, 'r') as f_untrim_ids, open(trim_ids_path, 'r') as f_trim_ids:
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

    act_cnames = sorted(set(act_cnames))
    sact_cnames = sorted(set(sact_cnames))
    actor_cnames = sorted(set(actor_cnames))
    object_cnames = sorted(set(object_cnames))
    aact_cnames = sorted(set(aact_cnames))
    relat_cnames = sorted(set(relat_cnames))

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

    print('Generate MOMA API indices')
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

    with open(act_cids_path, 'w') as f_act_cids, open(sact_cids_path, 'w') as f_sact_cids:
      f_act_cids.write(json.dumps(act_cids))
      f_sact_cids.write(json.dumps(sact_cids))

    with open(untrim_ids_path, 'w') as f_untrim_ids, open(trim_ids_path, 'w') as f_trim_ids:
      f_untrim_ids.write(json.dumps(untrim_ids))
      f_trim_ids.write(json.dumps(trim_ids))

    return act_cnames, sact_cnames, aact_cnames, \
           actor_cnames, object_cnames,  relat_cnames, \
           act_cids, sact_cids, untrim_ids, trim_ids

  @property
  def num_act_classes(self):
    return len(self.act_cnames)

  @property
  def num_sact_classes(self):
    return len(self.sact_cnames)

  @property
  def num_aact_classes(self):
    return len(self.aact_cnames)

  @property
  def num_actor_classes(self):
    return len(self.actor_cnames)

  @property
  def num_object_classes(self):
    return len(self.object_cnames)

  @property
  def num_relat_classes(self):
    return len(self.relat_cnames)

  @staticmethod
  def parse_bbox(raw_bbox: Dict, size: Size) -> BBox:
    x = [raw_bbox['topLeft']['x'], raw_bbox['bottomLeft']['x'], raw_bbox['topRight']['x'], raw_bbox['bottomRight']['x']]
    y = [raw_bbox['topLeft']['y'], raw_bbox['bottomLeft']['y'], raw_bbox['topRight']['y'], raw_bbox['bottomRight']['y']]

    x1 = max(round(min(x)), 0)
    y1 = max(round(min(y)), 0)
    w = min(round(max(x)-x1+1), size.w-1)
    h = min(round(max(y)-y1+1), size.h-1)

    return BBox(x1, y1, w, h)

  @staticmethod
  def parse_size(raw_size: dict) -> Size:
    return Size(raw_size['width'], raw_size['height'])

  def parse_actor(self, raw_actor: Dict, size: Size) -> Entity:
    cid = self.actor_cnames.index(raw_actor['class'])
    iid = raw_actor['id_in_video']
    bbox = self.parse_bbox(raw_actor['bbox'], size)

    return Entity(cid, iid, bbox)

  def parse_object(self, raw_object: Dict, size: Size) -> Entity:
    cid = self.object_cnames.index(raw_object['class'])
    iid = raw_object['id_in_video']
    bbox = self.parse_bbox(raw_object['bbox'], size)

    return Entity(cid, iid, bbox)

  def parse_relat(self, raw_relat: Dict) -> Relat:
    cname = raw_relat['class']
    description = raw_relat['description']

    cid = self.relat_cnames.index(cname)
    src_iids, snk_iids = description[1:-1].split('),(')
    src_iids = sorted(set(src_iids.split(',')))
    snk_iids = sorted(set(snk_iids.split(',')))

    return Relat(cid, src_iids, snk_iids)

  def parse_ag(self, raw_graph_ann):
    size = self.parse_size(raw_graph_ann['frame_dim'])
    actors = sorted(set([self.parse_actor(raw_actor, size) for raw_actor in raw_graph_ann['annotation']['actors']]))
    objects = sorted(set([self.parse_object(raw_object, size) for raw_object in raw_graph_ann['annotation']['objects']]))
    relats = set([self.parse_relat(raw_relat) for raw_relat in raw_graph_ann['annotation']['relationships']])
    ag = AG(actors, objects, relats)

    # sanity check
    actor_iids = sorted(set([actor['id_in_video'] for actor in raw_graph_ann['annotation']['actors']]))
    object_iids = sorted(set([object['id_in_video'] for object in raw_graph_ann['annotation']['objects']]))
    assert all([set(relat.src_iids).issubset(set(actor_iids+object_iids)) for relat in relats])
    assert all([set(relat.snk_iids).issubset(set(actor_iids+object_iids)) for relat in relats])

    return ag

  def parse_aact(self, raw_aact: List[List[Dict]], ags: List[AG]) -> AAct:
    assert len(raw_aact) == len(ags)
    num_frames = len(raw_aact)

    actor_iids = sorted(set(chain(*[ag.actor_iids for ag in ags])))
    aacts_actor_iids = sorted(set(chain(*[y['actor_id'].split(',') for x in raw_aact for y in x])))
    assert set(aacts_actor_iids).issubset(set(aacts_actor_iids))
    num_actors = len(actor_iids)
    encoded_tracklets = -2*np.ones((num_actors, num_frames), dtype=np.int64)  # absent

    actor_cids = {}
    for ag in ags:
      for actor_iid, actor_cid in zip(ag.actor_iids, ag.actor_cids):
        if actor_iid not in actor_cids:
          actor_cids[actor_iid] = actor_cid
    actor_cids = [actor_cids[actor_iid] for actor_iid in actor_iids]

    # encoded_aact = encoded_tracklets[i, j] is the encoded atomic action for actor actor_iids[i] in frame j
    for j, x in enumerate(raw_aact):
      # present
      j_pst_actor_iids = ags[j].actor_iids
      j_pst_actor_indices = [actor_iids.index(actor_iid) for actor_iid in j_pst_actor_iids]
      encoded_tracklets[j_pst_actor_indices, j] = -1

      # present + active
      j_aact_cids = [self.aact_cnames.index(y['class']) for y in x]
      j_pst_atv_actor_iids_list = [y['actor_id'].split(',') for y in x]

      for j_aact_cid, j_pst_atv_actor_iids in zip(j_aact_cids, j_pst_atv_actor_iids_list):
        j_pst_atv_actor_indices = [actor_iids.index(actor_iid) for actor_iid in j_pst_atv_actor_iids]
        for i in j_pst_atv_actor_indices:
          encoded_tracklets[i, j] = add_encoded_aact(encoded_tracklets[i, j], j_aact_cid)

    return AAct(actor_iids, actor_cids, encoded_tracklets, len(self.aact_cnames))
