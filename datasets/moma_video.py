from torchvision.datasets.video_utils import VideoClips
from torchvision.datasets.vision import VisionDataset

from .moma_api import get_moma_api


class MOMAVideo(VisionDataset):
  def __init__(self, cfg, transform):
    super(MOMAVideo, self).__init__(cfg.data_dir)

    self.cfg = cfg
    self.transform = transform
    self.api = get_moma_api(cfg.data_dir, 'trim')

    video_paths = []
    for trim_id in self.api.trim_ids:
      video_path = self.api.get_video_path(trim_id, False)
      video_paths.append(video_path)

    self.video_clips = VideoClips(video_paths,
                                  clip_length_in_frames=cfg.clip_length,
                                  frames_between_clips=cfg.clip_step,
                                  num_workers=cfg.num_workers)

  @property
  def metadata(self):
    return self.video_clips.metadata

  def __getitem__(self, index):
    video, _, info, video_index = self.video_clips.get_clip(index)
    trim_id = self.api.trim_ids[video_index]
    return trim_id, video

  def __len__(self):
    return self.video_clips.num_clips()
