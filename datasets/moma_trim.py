from torchvision import datasets, io
import torchvision.transforms.functional as F

from .moma_api import get_moma_api


class MOMATrim(datasets.VisionDataset):
  def __init__(self, cfg, fetch=()):
    super(MOMATrim, self).__init__(cfg.data_dir)

    self.cfg = cfg
    self.fetch = fetch
    self.api = get_moma_api(cfg.data_dir, 'trim')

    # cfg.num_actor_classes = len(self.api.actor_cnames)
    # cfg.num_object_classes = len(self.api.object_cnames)
    # cfg.num_relat_classes = len(self.api.relat_cnames)

  @staticmethod
  def resize(video, trim_ann, scale=0.5):
    h, w = video.shape[-2:]
    video = F.resize(video, [round(h*scale), round(w*scale)])
    trim_ann['ags'] = [ag.resize(scale) for ag in trim_ann['ags']]

    return video, trim_ann

  def __getitem__(self, index):
    trim_id = self.api.trim_ids[index]
    trim_ann = self.api.get_ann(trim_id)
    out = [trim_id, trim_ann]

    if 'video' in self.fetch:
      video_path = self.api.get_video_path(trim_id, True)
      video = io.read_video(video_path, pts_unit='sec')[0]
      video = video.float().permute(0, 3, 1, 2)/255.  # to_tensor

      # resize when width > 2000
      if video.shape[-1] > 2000:
        video, trim_ann = self.resize(video, trim_ann)
        out[1] = trim_ann

      out.append(video)
    elif 'feat' in self.fetch:
      pass

    return tuple(out)

  def __len__(self):
    return len(self.api.trim_ids)
