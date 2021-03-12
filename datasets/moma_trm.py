from torchvision import datasets, io

from .moma_api import get_moma_api


class MomaTrm(datasets.VisionDataset):
  def __init__(self, cfg):
    super(MomaTrm, self).__init__(cfg.data_dir)

    self.cfg = cfg
    self.api = get_moma_api(cfg.data_dir, 'trm')
    self.keys = sorted(self.api.anns.keys())

    # cfg.num_actor_classes = len(self.api.actor_cnames)
    # cfg.num_object_classes = len(self.api.object_cnames)
    # cfg.num_relationship_classes = len(self.api.relationship_cnames)

  def __getitem__(self, index):
    trm_id = self.keys[index]
    trm_ann = self.api.anns[trm_id]
    return trm_id, trm_ann

  def __len__(self):
    return len(self.api.anns)
