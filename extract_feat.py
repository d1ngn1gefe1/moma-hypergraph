import argparse
import torchvision.models as models

import datasets
import engine


parser = argparse.ArgumentParser()

parser.add_argument('--gpu', default=1, type=int)
parser.add_argument('--batch_size', default=20, type=int)
parser.add_argument('--data_dir', default='/home/ubuntu/datasets/MOMA', type=str)


def main():
  cfg = parser.parse_args()
  model = models.resnet152(pretrained=True)
  dataset = datasets.MomaTrm(cfg)
  feat_extractor = engine.FeatExtractor(cfg)
  feat_extractor.fit(model, dataset)


if __name__ == '__main__':
  main()
