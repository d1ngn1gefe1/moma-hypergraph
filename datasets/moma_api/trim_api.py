import os

from .base_api import BaseAPI


class TrimAPI(BaseAPI):
  def __init__(self, data_dir):
    super().__init__(data_dir)
    self._anns = self.load_anns()
    self.trim_ids = sorted(self._anns.keys())

  def load_anns(self):
    """
    Return
      - anns: a dict trim_id -> {
        size: Size
        fps: float
        ags: List[AG]
      }
    """
    anns = {}

    for raw_graph_ann in self.raw_graph_anns:
      trim_id = raw_graph_ann['trim_video_id']
      ag = self.parse_ag(raw_graph_ann)

      if trim_id not in anns:
        anns[trim_id] = {
          'size': self.parse_size(raw_graph_ann['frame_dim']),
          'fps': raw_graph_ann['fps'],
          'ags': [ag]
        }
      else:
        anns[trim_id]['ags'].append(ag)

    return anns

  def get_ann(self, trim_id):
    return self._anns[trim_id]

  def get_video_path(self, trim_id, sample):
    if sample:
      video_path = os.path.join(self.trim_sample_dir, '{}.mp4'.format(trim_id))
    else:
      video_path = os.path.join(self.trim_dir, '{}.mp4'.format(trim_id))

    return video_path
