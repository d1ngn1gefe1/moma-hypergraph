import torch
from torch.utils.data import DataLoader

import utils


class Trainer:
  def __init__(self, cfg):
    self.cfg = cfg
    self.device = torch.device('cuda:{}'.format(self.cfg.gpu) if torch.cuda.is_available() else 'cpu')
    self.logger = utils.Logger(self.cfg.save_dir, cfg)

  def fit(self, model, dataset_train, dataset_val):
    model = model.to(self.device)
    dataloader_train = DataLoader(dataset_train, batch_size=self.cfg.batch_size, shuffle=True,
                                  collate_fn=utils.collate_fn)
    dataloader_val = DataLoader(dataset_val, batch_size=self.cfg.batch_size, shuffle=False,
                                collate_fn=utils.collate_fn)
    optimizer = model.get_optimizer()

    for epoch in range(self.cfg.num_epochs):
      model.train()
      for i, batch in enumerate(dataloader_train):
        batch = batch.to(self.device)
        loss, acc = model(batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        stats = {'loss': loss.item(), 'acc': acc.item()}
        self.logger.update(batch.num_graphs, stats, 'train')

      model.eval()
      with torch.no_grad():
        for i, batch in enumerate(dataloader_val):
          batch = batch.to(self.device)
          loss, acc = model(batch)

          stats = {'loss': loss.item(), 'acc': acc.item()}
          self.logger.update(batch.num_graphs, stats, 'val')

      self.logger.summarize(epoch)
