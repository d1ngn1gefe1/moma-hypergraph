import argparse

import datasets
import engine
import models


parser = argparse.ArgumentParser()

parser.add_argument('--gpu', default=1, type=int)
parser.add_argument('--num_workers', default=4, type=int)

parser.add_argument('--data_dir', default='/home/ubuntu/datasets/MOMA', type=str)
parser.add_argument('--save_dir', default='/home/ubuntu/ckpt/moma-model', type=str)
parser.add_argument('--num_epochs', default=100, type=int)
parser.add_argument('--batch_size', default=4, type=int)

parser.add_argument('--lr', default=0.01, type=float)
parser.add_argument('--weight_decay', default=5e-4, type=float)


def main():
  cfg = parser.parse_args()

  dataset_train = datasets.MOMATrim(cfg, fetch=('feat',))
  # dataset_val = datasets.MOMATrim(cfg, fetch=('feat',))
  # model = models.RGCNModel(cfg)
  # trainer = engine.Trainer(cfg)
  #
  # trainer.fit(model, dataset_train, dataset_val)


if __name__ == '__main__':
  main()
