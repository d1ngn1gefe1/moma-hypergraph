import os

from .base_api import BaseAPI


class TrimAPI(BaseAPI):
  def __init__(self, data_dir):
    super().__init__(data_dir)
    self._trim_anns = self.load_anns()
    self.trim_ids = sorted(self._trim_anns.keys())

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

  def get_ann(self, trim_id):
    return self._trim_anns[trim_id]

  def get_video_path(self, trim_id, sample):
    if sample:
      video_path = os.path.join(self.trim_sample_dir, '{}.mp4'.format(trim_id))
    else:
      video_path = os.path.join(self.trim_dir, '{}.mp4'.format(trim_id))

    return video_path
