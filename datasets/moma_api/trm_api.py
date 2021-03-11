import os

from .base_api import BaseAPI


class TrmAPI(BaseAPI):
  def __init__(self, data_dir):
    super().__init__(data_dir)
    self.anns = self.load_anns()

  def load_anns(self):
    """
    Return
      - anns: a dict trm_id -> {
        size: Size
        fps: float
        ahgs: List[AHG]
      }
    """
    anns = {}

    for raw_graph_ann in self.raw_graph_anns:
      trm_id = raw_graph_ann['trim_video_id']
      ahg = self.parse_ahg(raw_graph_ann)

      if trm_id not in anns:
        anns[trm_id] = {
          'size': self.parse_size(raw_graph_ann['frame_dim']),
          'fps': raw_graph_ann['fps'],
          'ahgs': [ahg]
        }
      else:
        anns[trm_id]['ahgs'].append(ahg)

    return anns

  def get_ann(self, trm_id):
    return self.anns[trm_id]

  def get_video_path(self, trm_id):
    video_path = os.path.join(self.trm_smp_dir, '{}.mp4'.format(trm_id))

    return video_path
