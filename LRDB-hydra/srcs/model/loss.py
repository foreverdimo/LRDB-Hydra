import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

eps = 1e-8

class IQALoss(torch.nn.Module):
    def __init__(self, loss_type):
        super(IQALoss, self).__init__()
        self.loss_type = loss_type
    
    def forward(self, y_pred, y):
        if self.loss_type == 'l1':
            return F.l1_loss(y_pred, y)
        elif self.loss_type == 'l2':
            return F.mse_loss(y_pred, y)
        elif self.loss_type == 'norm-in-norm':
            return norm_loss_with_normalization(y_pred, y)
        else:
            raise ValueError


def norm_loss_with_normalization(y_pred, y, alpha=[1, 1], p=2, q=2, detach=False, exponent=True):
    """norm_loss_with_normalization: norm-in-norm, from https://github.com/lidq92/LinearityIQA"""
    N = y_pred.size(0)
    if N > 1:  
        m_hat = torch.mean(y_pred.detach()) if detach else torch.mean(y_pred)
        y_pred = y_pred - m_hat  # very important!!
        normalization = torch.norm(y_pred.detach(), p=q) if detach else torch.norm(y_pred, p=q)  # Actually, z-score normalization is related to q = 2.
        # print('bhat = {}'.format(normalization.item()))
        y_pred = y_pred / (eps + normalization)  # very important!
        y = y - torch.mean(y)
        y = y / (eps + torch.norm(y, p=q))
        scale = np.power(2, max(1,1./q)) * np.power(N, max(0,1./p-1./q)) # p, q>0
        loss0, loss1 = 0, 0
        if alpha[0] > 0:
            err = y_pred - y
            if p < 1:  # avoid gradient explosion when 0<=p<1; and avoid vanishing gradient problem when p < 0
                err += eps 
            loss0 = torch.norm(err, p=p) / scale  # Actually, p=q=2 is related to PLCC
            loss0 = torch.pow(loss0, p) if exponent else loss0 #
        if alpha[1] > 0:
            rho =  torch.cosine_similarity(y_pred, y)  #  
            err = rho * y_pred - y
            if p < 1:  # avoid gradient explosion when 0<=p<1; and avoid vanishing gradient problem when p < 0
                err += eps 
            loss1 = torch.norm(err, p=p) / scale  # Actually, p=q=2 is related to LSR
            loss1 = torch.pow(loss1, p) if exponent else loss1 #  #  
        # by = normalization.detach()
        # e0 = err.detach().view(-1)
        # ones = torch.ones_like(e0)
        # yhat = y_pred.detach().view(-1)
        # g0 = torch.norm(e0, p=p) / torch.pow(torch.norm(e0, p=p) + eps, p) * torch.pow(torch.abs(e0), p-1) * e0 / (torch.abs(e0) + eps)
        # ga = -ones / N * torch.dot(g0, ones)
        # gg0 = torch.dot(g0, g0)
        # gga = torch.dot(g0+ga, g0+ga)
        # print("by: {} without a and b: {} with a: {}".format(normalization, gg0, gga))
        # gb = -torch.pow(torch.abs(yhat), q-1) * yhat / (torch.abs(yhat) + eps) * torch.dot(g0, yhat)
        # gab = torch.dot(ones, torch.pow(torch.abs(yhat), q-1) * yhat / (torch.abs(yhat) + eps)) / N * torch.dot(g0, yhat)
        # ggb = torch.dot(g0+gb, g0+gb)
        # ggab = torch.dot(g0+ga+gb+gab, g0+ga+gb+gab)
        # print("by: {} without a and b: {} with a: {} with b: {} with a and b: {}".format(normalization, gg0, gga, ggb, ggab))
        return (alpha[0] * loss0 + alpha[1] * loss1) / (alpha[0] + alpha[1])
    else:
        return F.l1_loss(y_pred, y_pred.detach())  # 0 for batch with single sample.