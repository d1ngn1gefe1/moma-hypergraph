from .untrm_api import UntrmAPI
from .trm_api import TrmAPI
from .frm_api import FrmAPI


def get_moma_api(data_dir, level):
  """
  Three levels of data hierarchy:
   - Untrm-video-level
   - Trm-video-level
   - Frame-level

  Parameters:
    data_dir: the directory that contains videos and anns
    level: ['untrm', 'trm', 'frm']
  """
  assert level in ['untrm', 'trm', 'frm']

  if level == 'untrm':
    return UntrmAPI(data_dir)
  elif level == 'trm':
    return TrmAPI(data_dir)
  else:  # level == 'frm'
    return FrmAPI(data_dir)
