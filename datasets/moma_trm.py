from torchvision import datasets, io

from .moma_api import get_moma_api


class MomaTrm(datasets.VisionDataset):
  def __init__(self, cfg):
    self.cfg = cfg
    self.api = get_moma_api(cfg.data_dir, 'trm')
    self.keys = sorted(self.api.anns.keys())

    # cfg.num_actor_classes = len(self.api.actor_cnames)
    # cfg.num_object_classes = len(self.api.object_cnames)
    # cfg.num_relationship_classes = len(self.api.relationship_cnames)

    super(MomaTrm, self).__init__(cfg.data_dir)

  def __get_item__(self, index):
    trm_id = self.keys[index]
    trm_ann = self.api.anns[trm_id]

    video_path = self.api.get_video_path(trm_id)
    video = io.read_video(video_path, pts_unit='sec')[0]

    return video, trm_ann

  def __len__(self):
    return len(self.api.anns)
