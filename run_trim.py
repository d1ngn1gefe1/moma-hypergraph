import argparse
import datasets
import engine
import models


parser = argparse.ArgumentParser()

parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--num_workers', default=16, type=int)

parser.add_argument('--data_dir', default='/home/ubuntu/datasets/MOMA', type=str)
parser.add_argument('--save_dir', default='/home/ubuntu/ckpt/moma-model', type=str)
parser.add_argument('--num_epochs', default=300, type=int)
parser.add_argument('--batch_size', default=32, type=int)

parser.add_argument('--lr', default=5e-3, type=float)
parser.add_argument('--weight_decay', default=5e-4, type=float)

parser.add_argument('--split', default='split_by_untrim', type=str, choices=['split_by_trim', 'split_by_untrim'])
parser.add_argument('--backbone', default='HGCN', type=str, choices=['GIN', 'HGCN'])
parser.add_argument('--weights', default=[1.0, 1.0, 5.0, 5.0, 1.0], nargs='+', type=float,
                    help='act, sact, ps_aact, pa_aact, actor')
parser.add_argument('--tasks', default=['ps_aact', 'pa_aact'], nargs='+', type=str,
                    choices=['act', 'sact', 'ps_aact', 'pa_aact', 'actor'])


def main():
  cfg = parser.parse_args()

  dataset_train = datasets.MOMATrim(cfg, 'train', fetch='pyg')
  dataset_val = datasets.MOMATrim(cfg, 'val', fetch='pyg')

  model = models.MultitaskModel(cfg)
  trainer = engine.Trainer(cfg)

  trainer.fit(model, dataset_train, dataset_val)


if __name__ == '__main__':
  main()
