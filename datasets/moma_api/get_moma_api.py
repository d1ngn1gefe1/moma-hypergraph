from .untrim_api import UntrimAPI
from .trim_api import TrimAPI
from .frame_api import FrameAPI


def get_moma_api(data_dir, level):
  """
  Three levels of data hierarchy:
   - Untrim-video-level
   - Trim-video-level
   - Frame-level

  Parameters:
    data_dir: the directory that contains videos and anns
    level: ['untrim', 'trim', 'frame']
  """
  assert level in ['untrim', 'trim', 'frame']

  if level == 'untrim':
    return UntrimAPI(data_dir)
  elif level == 'trim':
    return TrimAPI(data_dir)
  else:  # level == 'frame'
    return FrameAPI(data_dir)
