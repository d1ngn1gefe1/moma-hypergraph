import argparse
import datasets
import engine
import models


parser = argparse.ArgumentParser()

# hardware
parser.add_argument('--gpu', default=3, type=int)
parser.add_argument('--num_workers', default=16, type=int)

# file system
parser.add_argument('--data_dir', default='/home/ubuntu/datasets/MOMA', type=str)
parser.add_argument('--save_dir', default='/home/ubuntu/ckpt/moma-model', type=str)
parser.add_argument('--feats_dname', default='feats_18', type=str)
parser.add_argument('--split_by', default='untrim', type=str, choices=['trim', 'untrim'])

# hyper-parameters
parser.add_argument('--num_epochs', default=100, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=5e-3, type=float)
parser.add_argument('--weight_decay', default=5e-4, type=float)

# experiments
parser.add_argument('--backbone', default='GCN', type=str, choices=['GINE', 'GCN', 'HGCN'],
                    help='GINE usess edge attr, GCN and HGCN do not')
parser.add_argument('--oracle', default=False, action='store_true')
parser.add_argument('--weights', default=[1.0, 1.0, 5.0, 5.0, 1.0, 1.0], nargs='+', type=float,
                    help='act, sact, ps_aact, pa_aact, actor, relat')
parser.add_argument('--tasks', default=['act', 'sact', 'ps_aact', 'pa_aact', 'actor', 'relat'], nargs='+', type=str,
                    choices=['act', 'sact', 'ps_aact', 'pa_aact', 'actor', 'relat'])


def main():
  cfg = parser.parse_args()

  dataset_train = datasets.MOMATrim(cfg, split='train', fetch='pyg')
  dataset_val = datasets.MOMATrim(cfg, split='val', fetch='pyg')

  model = models.MultitaskModel(cfg)
  trainer = engine.Trainer(cfg)

  trainer.fit(model, dataset_train, dataset_val)


if __name__ == '__main__':
  main()
