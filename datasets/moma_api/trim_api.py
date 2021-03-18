import os

from .base_api import BaseAPI


class TrimAPI(BaseAPI):
  def __init__(self, data_dir, split_by, feats_dname):
    super().__init__(data_dir, split_by=split_by, feats_dname=feats_dname)
    self._trim_anns = self.load_anns()

  def load_anns(self):
    """
    Return
      - trim_anns: a dict trim_id -> {
        size: Size
        fps: float
        ags: List[AG]
        aact: AAct
      }
    """
    trim_anns = {}
    raw_aacts = {}

    for raw_graph_ann in self.raw_graph_anns:
      trim_id = raw_graph_ann['trim_video_id']
      ag = self.parse_ag(raw_graph_ann)
      raw_aact = raw_graph_ann['annotation']['atomic_actions']

      if ag.is_empty:
        continue

      if trim_id not in trim_anns:  # trim_id not in raw_aacts
        trim_anns[trim_id] = {
          'size': self.parse_size(raw_graph_ann['frame_dim']),
          'fps': raw_graph_ann['fps'],
          'ags': [ag]
        }
        raw_aacts[trim_id] = [raw_aact]
      else:
        trim_anns[trim_id]['ags'].append(ag)
        raw_aacts[trim_id].append(raw_aact)

    # parse atomic actions
    for trim_id in list(trim_anns.keys()):
      trim_anns[trim_id]['aact'] = self.parse_aact(raw_aacts[trim_id], trim_anns[trim_id]['ags'])

    return trim_anns

  def get_trim_ids(self, split=None):
    if split is None:
      return sorted(self._trim_anns.keys())
    elif split == 'train':
      return self.split_train
    elif split == 'val':
      return self.split_val
    else:
      raise ValueError

  def get_ann(self, trim_id):
    return self._trim_anns[trim_id]

  def get_video_path(self, trim_id, sample):
    if sample:
      video_path = os.path.join(self.trim_sample_dir, '{}.mp4'.format(trim_id))
    else:
      video_path = os.path.join(self.trim_dir, '{}.mp4'.format(trim_id))

    return video_path
