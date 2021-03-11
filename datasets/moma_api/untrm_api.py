from .base_api import BaseAPI


class UntrmAPI(BaseAPI):
  def __init__(self, data_dir):
    super().__init__(data_dir)
    self.anns = self.load_anns()

  def load_anns(self):
    raise NotImplementedError

  def get_ann(self, untrm_id):
    return self.anns[untrm_id]

  def get_video_path(self):
    raise NotImplementedError
