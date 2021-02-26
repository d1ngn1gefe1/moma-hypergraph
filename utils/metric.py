import torch


def accuracy(logits, labels):
  with torch.no_grad():
    batch_size = labels.size(0)
    _, predicts = logits.max(1)
    correct = predicts.eq(labels).sum()
    acc = 100.*correct/batch_size

    return acc
