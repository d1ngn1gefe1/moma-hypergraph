from .base_api import BaseAPI


class UntrimAPI(BaseAPI):
  def __init__(self, data_dir):
    super().__init__(data_dir)
    self.anns = self.load_anns()

  def load_anns(self):
    raise NotImplementedError

  def get_ann(self, untrim_id):
    return self.anns[untrim_id]

  def get_video_path(self):
    raise NotImplementedError
