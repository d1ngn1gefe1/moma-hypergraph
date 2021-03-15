from .untrim_api import UntrimAPI
from .trim_api import TrimAPI
from .frame_api import FrameAPI


def get_moma_api(data_dir, split, level):
  """
  Three levels of data hierarchy:
   - Untrim-video-level
   - Trim-video-level
   - Frame-level

  Parameters:
    data_dir: the directory that contains videos and anns
    split: ['split_by_trim', 'split_by_untrim']
    level: ['untrim', 'trim', 'frame']
  """
  assert split in ['split_by_trim', 'split_by_untrim']
  assert level in ['untrim', 'trim', 'frame']

  if level == 'untrim':
    return UntrimAPI(data_dir)
  elif level == 'trim':
    return TrimAPI(data_dir, split)
  else:  # level == 'frame'
    return FrameAPI(data_dir)
