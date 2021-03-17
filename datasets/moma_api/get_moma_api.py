from .untrim_api import UntrimAPI
from .trim_api import TrimAPI
from .frame_api import FrameAPI


def get_moma_api(data_dir, split_by, feats_dname, level):
  """
  Three levels of data hierarchy:
   - Untrim-video-level
   - Trim-video-level
   - Frame-level

  Parameters:
    data_dir: the directory that contains videos and anns
    split_by: ['trim', 'untrim']
    feats_dname: features directory name
    level: ['untrim', 'trim', 'frame']
  """
  assert split_by in ['trim', 'untrim'], split_by
  assert level in ['untrim', 'trim', 'frame'], level

  if level == 'untrim':
    return UntrimAPI(data_dir)
  elif level == 'trim':
    return TrimAPI(data_dir, split_by, feats_dname)
  else:  # level == 'frame'
    return FrameAPI(data_dir)
