from abc import ABC
import json
import math
import os


def get_momaapi(dataset_dir, level, sampled=True):
  """
  Three levels of hierarchy:
   - Untrimmed video
   - Trimmed video
   - Spatial graph

  Parameters:
    dataset_dir: the directory that contains videos and annotations
    level: ['untrimmed_video', 'trimmed_video', 'spatial_graph']
    sampled: for MomaApiTrimmedVideo only, whether to load the sampled videos
  """
  assert level in ['untrimmed_video', 'trimmed_video', 'spatial_graph']

  if level == 'untrimmed_video':
    return MomaApiUntrimmedVideo(dataset_dir)
  elif level == 'trimmed_video':
    return MomaApiTrimmedVideo(dataset_dir, sampled=sampled)
  else:  # level == 'spatial_graph'
    return MomaApiSpatialGraph(dataset_dir)


class MomaApi(ABC):
  def __init__(self, dataset_dir, annotations_dname='annotations', untrimmed_videos_dname='untrimmed_videos',
               trimmed_videos_dname='trimmed_videos', trimmed_sampled_videos_dname='trimmed_sampled_videos'):
    self.annotations_dir = os.path.join(dataset_dir, annotations_dname)
    self.untrimmed_videos_dir = os.path.join(dataset_dir, untrimmed_videos_dname)
    self.trimmed_videos_dir = os.path.join(dataset_dir, trimmed_videos_dname)
    self.trimmed_sampled_videos_dir = os.path.join(dataset_dir, trimmed_sampled_videos_dname)

    self.raw_video_annotations = self.load_raw_video_annotations()
    self.raw_graph_annotations = self.load_raw_graph_annotations()

    self.activity_cnames, self.subactivity_cnames, self.actor_cnames, self.object_cnames, \
        self.atomic_action_cnames, self.relationship_cnames = self.get_cnames()
    self.__trimmed_to_untrimmed = self.setup_mapping()

  def load_raw_video_annotations(self):
    """
    Load video annotations
    """
    fname = 'video.json'

    with open(os.path.join(self.annotations_dir, fname), 'r') as f:
      raw_video_annotations = json.load(f)

    return raw_video_annotations

  def load_raw_graph_annotations(self):
    """
    Load graph annotations
    """
    fnames = ['graph_0_10000.json', 'graph_10000_20000.json', 'graph_20000_30000.json', 'graph_30000_37735.json']

    raw_graph_annotations = []
    for fname in fnames:
      with open(os.path.join(self.annotations_dir, fname), 'r') as f:
        raw_graph_annotations += json.load(f)

    return raw_graph_annotations

  def get_cnames(self):
    """
    Return a list of arrays that map from class id to class name
      - activity_cnames: activity_cid -> activity_cname
      - subactivity_cnames: subactivity_cid -> subactivity_cname
      - actor_cnames: actor_cid -> actor_cname
      - object_cnames: object_cid -> object_cname
      - atomic_action_cnames: atomic_action_cid -> atomic_action_cname
      - relationship_cnames: relationship_cid -> relationship_cname
    """
    activity_cnames_path = os.path.join(self.annotations_dir, 'activity_cnames.txt')
    subactivity_cnames_path = os.path.join(self.annotations_dir, 'subactivity_cnames.txt')
    actor_cnames_path = os.path.join(self.annotations_dir, 'actor_cnames.txt')
    object_cnames_path = os.path.join(self.annotations_dir, 'object_cnames.txt')
    atomic_action_cnames_path = os.path.join(self.annotations_dir, 'atomic_action_cnames.txt')
    relationship_cnames_path = os.path.join(self.annotations_dir, 'relationship_cnames.txt')

    if all(list(map(os.path.isfile, [activity_cnames_path, subactivity_cnames_path, actor_cnames_path,
                                     object_cnames_path, atomic_action_cnames_path, relationship_cnames_path]))):
      print('Load cnames')
      with open(activity_cnames_path, 'r') as f:
        activity_cnames = f.read().splitlines()
      with open(subactivity_cnames_path, 'r') as f:
        subactivity_cnames = f.read().splitlines()
      with open(actor_cnames_path, 'r') as f:
        actor_cnames = f.read().splitlines()
      with open(object_cnames_path, 'r') as f:
        object_cnames = f.read().splitlines()
      with open(atomic_action_cnames_path, 'r') as f:
        atomic_action_cnames = f.read().splitlines()
      with open(relationship_cnames_path, 'r') as f:
        relationship_cnames = f.read().splitlines()

      return activity_cnames, subactivity_cnames, actor_cnames, object_cnames, atomic_action_cnames, relationship_cnames

    activity_cnames, subactivity_cnames = [], []
    for untrimmed_video_id, raw_video_annotation in self.raw_video_annotations.items():
      activity_cnames.append(raw_video_annotation['class'])
      for subactivity in raw_video_annotation['subactivity']:
        subactivity_cnames.append(subactivity['class'])

    actor_cnames, object_cnames, atomic_action_cnames, relationship_cnames = [], [], [], []
    for raw_graph_annotation in self.raw_graph_annotations:
      actor_cnames += [actor['class'] for actor in
                       raw_graph_annotation['annotation']['characters']]
      object_cnames += [object['class'] for object in
                        raw_graph_annotation['annotation']['objects']]
      atomic_action_cnames += [atomic_action['class'] for atomic_action in
                               raw_graph_annotation['annotation']['atomic_actions']]
      relationship_cnames += [relationship['class'] for relationship in
                              raw_graph_annotation['annotation']['relationships']]

    activity_cnames = sorted(list(set(activity_cnames)))
    subactivity_cnames = sorted(list(set(subactivity_cnames)))
    actor_cnames = sorted(list(set(actor_cnames)))
    object_cnames = sorted(list(set(object_cnames)))
    atomic_action_cnames = sorted(list(set(atomic_action_cnames)))
    relationship_cnames = sorted(list(set(relationship_cnames)))

    print('Generate cnames')
    with open(activity_cnames_path, 'w') as f:
      f.write('\n'.join(activity_cnames))
    with open(subactivity_cnames_path, 'w') as f:
      f.write('\n'.join(subactivity_cnames))
    with open(actor_cnames_path, 'w') as f:
      f.write('\n'.join(actor_cnames))
    with open(object_cnames_path, 'w') as f:
      f.write('\n'.join(object_cnames))
    with open(atomic_action_cnames_path, 'w') as f:
      f.write('\n'.join(atomic_action_cnames))
    with open(relationship_cnames_path, 'w') as f:
      f.write('\n'.join(relationship_cnames))

    return activity_cnames, subactivity_cnames, actor_cnames, object_cnames, atomic_action_cnames, relationship_cnames

  def setup_mapping(self):
    """
    Setup a mapping dict necessary for the function to_untrimmed_video_id()
    """
    trimmed_to_untrimmed = {}
    for untrimmed_video_id, raw_video_annotation in self.raw_video_annotations.items():
      for subactivity in raw_video_annotation['subactivity']:
        trimmed_to_untrimmed[subactivity['trim_video_id']] = untrimmed_video_id

    return trimmed_to_untrimmed

  @staticmethod
  def get_spatial_graph_id(raw_graph_annotation):
    return '{}_{}'.format(raw_graph_annotation['instance_id'], raw_graph_annotation['picture_num'])

  @staticmethod
  def parse_spatial_graph_id(spatial_graph_id):
    trimmed_video_id, frame_id = spatial_graph_id.split('_')
    return trimmed_video_id, frame_id

  def to_untrimmed_video_id(self, trimmed_video_id):
    """
    Return the untrimmed video id given the trimmed video id
    """
    untrimmed_video_id = self.__trimmed_to_untrimmed[trimmed_video_id]
    return untrimmed_video_id

  def to_trimmed_video_id(self, spatial_graph_id):
    """
    Return the trimmed video id given the spatial graph
    """
    trimmed_video_id, _ = self.parse_spatial_graph_id(spatial_graph_id)
    return trimmed_video_id

  @staticmethod
  def parse_bbox(bbox):
    """
    Return the bounding box as [x1, y1, w, h]
    """
    x = [bbox['topLeft']['x'], bbox['bottomLeft']['x'], bbox['topRight']['x'], bbox['bottomRight']['x']]
    y = [bbox['topLeft']['y'], bbox['bottomLeft']['y'], bbox['topRight']['y'], bbox['bottomRight']['y']]

    x1 = min(x)
    y1 = min(y)
    w = max(x)-x1+1
    h = max(y)-y1+1

    return [x1, y1, w, h]

  @staticmethod
  def parse_size(size):
    """
    Return the video size as [w, h]
    """
    return [size['width'], size['height']]

  @staticmethod
  def parse_actor_instance_ids(actor_instance_ids):
    return actor_instance_ids.split(',')

  @staticmethod
  def parse_description(description):
    """
    Return description as a list
    """
    description = description[1:-1].split('),(')
    description = [d.split(',') for d in description]
    return description


class MomaApiUntrimmedVideo(MomaApi):
  def __init__(self, dataset_dir):
    super().__init__(dataset_dir)
    self.annotations = self.load_annotations()

  def load_annotations(self):
    """
    Return
      - annotations: a dict untrimmed_video_id -> {
          fps: int,
          size: [width, height],
          activity_cid: int,
          subactivities: a list of {
            trimmed_video_id: str,
            subactivity_cid: int,
            start: float,
            end: float
          }
        }
    """
    annotations = {}

    for untrimmed_video_id, raw_video_annotation in self.raw_video_annotations.items():
      assert untrimmed_video_id == raw_video_annotation['video_id']

      annotation = {
        'activity_cid': self.activity_cnames.index(raw_video_annotation['class']),
        'subactivities': [{'trimmed_video_id': subactivity['trim_video_id'],
                           'subactivity_cid': self.subactivity_cnames.index(subactivity['class']),
                           'start': subactivity['start'],
                           'end': subactivity['end']} for subactivity in raw_video_annotation['subactivity']]
      }

      annotations[untrimmed_video_id] = annotation

    for raw_graph_annotation in self.raw_graph_annotations:
      trimmed_video_id = raw_graph_annotation['instance_id']
      untrimmed_video_id = self.to_untrimmed_video_id(trimmed_video_id)

      if 'fps' not in annotations[untrimmed_video_id]:
        annotations[untrimmed_video_id]['fps'] = raw_graph_annotation['fps']

      if 'size' not in annotations[untrimmed_video_id]:
        annotations[untrimmed_video_id]['fps'] = self.parse_size(raw_graph_annotation['video_size'])

    return annotations

  def get_annotation(self, untrimmed_video_id):
    return self.annotations[untrimmed_video_id]

  def get_video_path(self):
    raise NotImplementedError


class MomaApiTrimmedVideo(MomaApi):
  def __init__(self, dataset_dir, sampled):
    super().__init__(dataset_dir)
    self.annotations = self.load_annotations()
    self.sampled = sampled

  def load_annotations(self):
    """
    Return
      - annotations: a dict trimmed_video_id -> {
          fps: int,
          size: [width, height],
          activity_cid: int,
          subactivity_cid: int
          graphs: an in-order list of {
            actors: a list of {
              actor_cid: int (25 actor classes),
              bbox: [x1, y1, w, h],
              instance_id: str
            },
            objects: a list of {
              object_cid: int (197 object classes),
              instance_id: str
            },
            atomic_actions: a list of {
              atomic_action_cid: int (64 atomic action classes),
              actor_instance_ids: a list of str
            },
            relationships: a list of {
              relationship_cid: int (35 relationship classes),
              description: str
            }
          }
        }
    """
    annotations = {}
    for untrimmed_video_id, raw_video_annotation in self.raw_video_annotations.items():
      for subactivity in raw_video_annotation['subactivity']:
        annotation = {
          'activity_cid': self.activity_cnames.index(raw_video_annotation['class']),
          'subactivity_cid': self.subactivity_cnames.index(subactivity['class']),
          'graphs': []
        }
        annotations[subactivity['trim_video_id']] = annotation

    for raw_graph_annotation in self.raw_graph_annotations:
      trimmed_video_id = raw_graph_annotation['instance_id']
      annotation = {
        'actors': [{
          'actor_cid': self.actor_cnames.index(actor['class']),
          'bbox': self.parse_bbox(actor['bbox']),
          'instance_id': actor['id_in_video']}
          for actor in raw_graph_annotation['annotation']['characters']],
        'objects': [{
          'object_cid': self.object_cnames.index(object['class']),
          'bbox': self.parse_bbox(object['bbox']),
          'instance_id': object['id_in_video']}
          for object in raw_graph_annotation['annotation']['objects']],
        'atomic_actions': [{
          'atomic_action_cid': self.atomic_action_cnames.index(atomic_action['class']),
          'actor_instance_ids': self.parse_actor_instance_ids(atomic_action['actor_id'])}
          for atomic_action in raw_graph_annotation['annotation']['atomic_actions']],
        'relationships': [{
          'relationship_cid': self.relationship_cnames.index(relationship['class']),
          'description': self.parse_description(relationship['description'])}
          for relationship in raw_graph_annotation['annotation']['relationships']]
      }
      assert len(annotations[trimmed_video_id]['graphs']) == raw_graph_annotation['picture_num']-1  # in-order
      annotations[trimmed_video_id]['graphs'].append(annotation)

      if 'fps' not in annotations[trimmed_video_id]:
        annotations[trimmed_video_id]['fps'] = raw_graph_annotation['fps']

      if 'size' not in annotations[trimmed_video_id]:
        annotations[trimmed_video_id]['size'] = self.parse_size(raw_graph_annotation['video_size'])

    return annotations

  def get_annotation(self, trimmed_video_id):
    return self.annotations[trimmed_video_id]

  def get_video_path(self, trimmed_video_id):
    if self.sampled:
      videos_dir = self.trimmed_sampled_videos_dir
    else:
      videos_dir = self.trimmed_videos_dir

    video_path = os.path.join(videos_dir, '{}.mp4'.format(trimmed_video_id))

    return video_path


class MomaApiSpatialGraph(MomaApi):
  def __init__(self, dataset_dir):
    super().__init__(dataset_dir)
    self.annotations = self.load_annotations()

  def load_annotations(self):
    """
    Return
      - annotations: a dict spatial_graph_id -> {
          activity_cid: int,
          subactivity_cid: int
          graph: {
            actors: a list of {
              actor_cid: int (25 actor classes),
              bbox: [x1, y1, w, h],
              instance_id: str
            },
            objects: a list of {
              object_cid: int (197 object classes),
              instance_id: str
            },
            atomic_actions: a list of {
              atomic_action_cid: int (64 atomic action classes),
              actor_instance_ids: a list of str
            },
            relationships: a list of {
              relationship_cid: int (35 relationship classes),
              description: str
            }
          }
        }
    """
    annotations = {}

    for raw_graph_annotation in self.raw_graph_annotations:
      spatial_graph_id = self.get_spatial_graph_id(raw_graph_annotation)
      annotation = {
        'actors': [{
          'actor_cid': self.actor_cnames.index(actor['class']),
          'bbox': self.parse_bbox(actor['bbox']),
          'instance_id': actor['id_in_video']}
          for actor in raw_graph_annotation['annotation']['characters']],
        'objects': [{
          'object_cid': self.object_cnames.index(object['class']),
          'bbox': self.parse_bbox(object['bbox']),
          'instance_id': object['id_in_video']}
          for object in raw_graph_annotation['annotation']['objects']],
        'atomic_actions': [{
          'atomic_action_cid': self.atomic_action_cnames.index(atomic_action['class']),
          'actor_instance_ids': self.parse_actor_instance_ids(atomic_action['actor_id'])}
          for atomic_action in raw_graph_annotation['annotation']['atomic_actions']],
        'relationships': [{
          'relationship_cid': self.relationship_cnames.index(relationship['class']),
          'description': self.parse_description(relationship['description'])}
          for relationship in raw_graph_annotation['annotation']['relationships']]
      }
      annotations[spatial_graph_id] = annotation

    return annotations

  def get_annotation(self, spatial_graph_id):
    return self.annotations[spatial_graph_id]

  def get_video_path(self):
    raise NotImplementedError
