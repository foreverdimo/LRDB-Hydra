import torch
import torch.nn.functional as F
import numpy as np
from scipy import stats
import math

def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)

def mse(output, target):
    with torch.no_grad():
        return F.mse_loss(output, target)

def srocc(output, target):
    sq = np.reshape(np.asarray(target), (-1,))
    q = np.reshape(np.asarray(output), (-1,))
    return stats.spearmanr(sq, q)[0]

def plcc(output, target):
    sq = np.reshape(np.asarray(target), (-1,))
    q = np.reshape(np.asarray(output), (-1,))
    return stats.pearsonr(sq, q)[0]