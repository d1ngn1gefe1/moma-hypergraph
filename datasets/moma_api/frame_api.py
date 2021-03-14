from .base_api import BaseAPI


class FrameAPI(BaseAPI):
  def __init__(self, data_dir):
    super().__init__(data_dir)
    self._anns = self.load_anns()

  def load_anns(self):
    """
    Return
      - anns: a dict frame_id -> {
        size: Size
        ag: AG
      }
    """
    anns = {}

    for raw_graph_ann in self.raw_graph_anns:
      frame_id = self.get_frame_id(raw_graph_ann)
      ag = self.parse_ag(raw_graph_ann)

      anns[frame_id] = {
        'size': self.parse_size(raw_graph_ann['frame_dim']),
        'ag': ag
      }

    return anns

  def get_frame_ids(self):
    return sorted(self._anns.keys())

  def get_ann(self, frame_id):
    return self._anns[frame_id]

  def get_video_path(self):
    raise NotImplementedError
