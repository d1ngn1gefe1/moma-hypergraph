import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score
import torch


def get_acc(logits, labels):
  with torch.no_grad():
    # single-label classification, labels: shape=[num_examples], range=[0, num_class)
    if logits.ndim == 2 and labels.ndim == 1:
      _, predicts = logits.max(1)

    # multi-label classification, labels: [num_examples, num_class], range=[0, 1]
    elif logits.ndim == labels.ndim == 2:
      # the set of labels and predictions must exactly match
      predicts = torch.round(torch.sigmoid(logits))

    else:
      raise NotImplementedError

    labels = labels.detach().cpu().numpy()
    predicts = predicts.detach().cpu().numpy()

    acc = accuracy_score(labels, predicts)
    return acc


def get_mAP(logits, labels):
  with torch.no_grad():
    labels = labels.detach().cpu().numpy()
    logits = logits.detach().cpu().numpy()

    # single-label classification, labels: shape=[num_examples], range=[0, num_class)
    if logits.ndim == 2 and labels.ndim == 1:
      indices = labels
      labels = np.zeros_like(logits)
      labels[np.arange(indices.shape[0]), indices] = 1

    # multi-label classification, labels: [num_examples, num_class], range=[0, 1]
    elif logits.ndim == labels.ndim == 2:
      pass

    else:
      raise NotImplementedError

    mAP = average_precision_score(labels, logits, average='micro')

    return mAP
