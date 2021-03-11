import os
from torch.utils.tensorboard import SummaryWriter


class AverageMeter:
  def __init__(self, name, fmt=':f'):
    self.name = name
    self.fmt = fmt

    self.val, self.avg, self.sum, self.count = None, None, None, None

    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val*n
    self.count += n
    self.avg = self.sum/self.count

  def __str__(self):
    fmtstr = '{name}: {avg'+self.fmt+'}'
    return fmtstr.format(**self.__dict__)


class Logger:
  def __init__(self, save_dir, cfg):
    self.writer = SummaryWriter(save_dir)
    self.meters = {}

    # save hparams
    f = open(os.path.join(save_dir, 'hparams.txt'), 'a')
    f.write(str(cfg))
    f.close()

  def __del__(self):
    self.writer.close()

  def update(self, n, stats, split):
    for key, value in stats.items():
      tag = '{}/{}'.format(key, split)

      if tag not in self.meters.keys():
        self.meters[tag] = AverageMeter(name=tag)

      self.meters[tag].update(value, n)

  def summarize(self, epoch, stats=None):
    print('---------- Epoch {} ----------'.format(epoch))

    for tag in self.meters.keys():
      self.writer.add_scalar(tag, self.meters[tag].avg, global_step=epoch)
      print(self.meters[tag])

    if stats is not None:
      for key, value in stats.items():
        self.writer.add_scalar(key, value, global_step=epoch)
        print('{}: {}'.format(key, value))

    # reset
    for tag in self.meters.keys():
      self.meters[tag].reset()