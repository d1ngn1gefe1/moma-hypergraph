import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score
import torch
import torch.nn.functional as F


def get_acc(logits, labels):
  with torch.no_grad():
    # single-label classification, labels: shape=[num_examples], range=[0, num_class)
    if logits.ndim == 2 and labels.ndim == 1:
      _, preds = logits.max(1)

    # multi-label classification, labels: [num_examples, num_class], range=[0, 1]
    elif logits.ndim == labels.ndim == 2:
      # the set of labels and predictions must exactly match
      preds = torch.round(torch.sigmoid(logits))

    else:
      raise NotImplementedError

    labels = labels.detach().cpu().numpy()
    preds = preds.detach().cpu().numpy()

    acc = accuracy_score(labels, preds)
    return acc


def get_map_slowfast(preds, labels):
  """ Reference: https://github.com/facebookresearch/SlowFast/blob/master/slowfast/utils/meters.py#L744
  Compute mAP for multi-label case.
  Args:
      preds (numpy tensor): num_examples x num_classes.
      labels (numpy tensor): num_examples x num_classes.
  Returns:
      mean_ap (int): final mAP score.
  """
  preds = preds[:, ~(np.all(labels == 0, axis=0))]
  labels = labels[:, ~(np.all(labels == 0, axis=0))]
  aps = [0]
  try:
    aps = average_precision_score(labels, preds, average=None)
  except ValueError:
    print(
      "Average precision requires a sufficient number of samples \
      in a batch which are missing in this sample."
    )

  mean_ap = np.mean(aps)
  return mean_ap


def get_mAP(logits, labels):
  with torch.no_grad():
    # single-label classification, labels: shape=[num_examples], range=[0, num_class)
    if logits.ndim == 2 and labels.ndim == 1:
      labels = labels.detach().cpu().numpy()
      probs = F.softmax(logits, dim=1).detach().cpu().numpy()

      indices = labels
      labels = np.zeros_like(probs)
      labels[np.arange(indices.shape[0]), indices] = 1

      mAP = average_precision_score(labels, probs, average='micro')

    # multi-label classification, labels: [num_examples, num_class], range=[0, 1]
    elif logits.ndim == labels.ndim == 2:
      labels = labels.detach().cpu().numpy()
      probs = torch.sigmoid(logits.detach()).cpu().numpy()
      mAP = get_map_slowfast(probs, labels)

    else:
      raise NotImplementedError

    return mAP
