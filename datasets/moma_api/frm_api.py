from .base_api import BaseAPI


class FrmAPI(BaseAPI):
  def __init__(self, data_dir):
    super().__init__(data_dir)
    self.anns = self.load_anns()

  def load_anns(self):
    """
    Return
      - anns: a dict frm_id -> {
        size: Size
        ahg: AHG
      }
    """
    anns = {}

    for raw_graph_ann in self.raw_graph_anns:
      frm_id = self.get_frm_id(raw_graph_ann)
      ahg = self.parse_ahg(raw_graph_ann)

      anns[frm_id] = {
        'size': self.parse_size(raw_graph_ann['frame_dim']),
        'ahg': ahg
      }

    return anns

  def get_annotation(self, frm_id):
    return self.anns[frm_id]

  def get_video_path(self):
    raise NotImplementedError
