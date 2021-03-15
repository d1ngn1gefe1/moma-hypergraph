import torch
from torch.utils.data import DataLoader

import utils


class Trainer:
  def __init__(self, cfg):
    self.cfg = cfg
    self.device = torch.device('cuda:{}'.format(cfg.gpu) if torch.cuda.is_available() else 'cpu')
    self.logger = utils.Logger(cfg.save_dir, cfg)

  def fit(self, model, dataset_train, dataset_val):
    dataloader_train = DataLoader(dataset_train, batch_size=self.cfg.batch_size, shuffle=True,
                                  collate_fn=utils.collate_fn)
    dataloader_val = DataLoader(dataset_val, batch_size=self.cfg.batch_size, shuffle=False,
                                collate_fn=utils.collate_fn)
    optimizer = model.get_optimizer()
    scheduler = model.get_scheduler(optimizer)

    model = model.to(self.device)

    for epoch in range(self.cfg.num_epochs):
      # train
      model.train()
      for i, batch in enumerate(dataloader_train):
        batch = batch.to(self.device)
        loss, acc = model(batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        stats_step = {'loss': loss.item(), 'acc': acc.item()}
        self.logger.update(batch.num_graphs, stats_step, 'train')

      # lr decay
      if scheduler is not None:
        scheduler.step()

      # val
      model.eval()
      with torch.no_grad():
        for i, batch in enumerate(dataloader_val):
          batch = batch.to(self.device)
          loss, acc = model(batch)

          stats_step = {'loss': loss.item(), 'acc': acc.item()}
          self.logger.update(batch.num_graphs, stats_step, 'val')

      stats_epoch = {'lr': optimizer.param_groups[0]['lr']}
      self.logger.summarize(epoch, stats=stats_epoch)
