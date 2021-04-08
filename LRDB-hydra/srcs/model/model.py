import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import models
from torchvision import transforms




class Backbone_Network(nn.Module):
    def __init__(self):
        super(Backbone_Network,self).__init__()
        self.CB1 = nn.Sequential(
            nn.Conv2d(in_channels= 1,out_channels= 16,kernel_size=(3,3),stride=(1,1)),
            nn.BatchNorm2d(num_features= 16),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels= 16,out_channels= 32,kernel_size=(3,3),stride=(2,2),padding=(2,2)),
            nn.BatchNorm2d(num_features= 32),
            nn.LeakyReLU()
        )
        self.CB2 = nn.Sequential(
            nn.Conv2d(in_channels= 1,out_channels= 16,kernel_size=(3,3),stride=(1,1),),
            nn.BatchNorm2d(num_features= 16),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels= 16,out_channels= 32,kernel_size=(3,3),stride=(2,2),padding=(2,2)),
            nn.BatchNorm2d(num_features= 32),
            nn.LeakyReLU()
        )
        self.CB3 = nn.Sequential(
            nn.Conv2d(in_channels= 64,out_channels= 64,kernel_size=(3,3),stride=(1,1)),
            nn.BatchNorm2d(num_features= 64),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels= 64,out_channels= 128,kernel_size=(3,3),stride=(2,2),padding=(2,2)),
            nn.BatchNorm2d(num_features= 128),
            nn.LeakyReLU()
        )
        self.CB4 = nn.Sequential(
            nn.Conv2d(in_channels= 128,out_channels= 256,kernel_size=(3,3),stride=(1,1)),
            nn.BatchNorm2d(num_features= 256),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels= 256,out_channels= 256,kernel_size=(3,3),stride=(2,2),padding=(2,2)),
            nn.BatchNorm2d(num_features= 256),
            nn.LeakyReLU()
        )
    def forward(self,data):
        (Id, E) = data
        Id = self.CB1(Id)
        E = self.CB2(E)
        x = torch.cat((Id,E),dim = 1)
        x = self.CB3(x)
        x = self.CB4(x)
        return x

class Local_QMG(nn.Module):
    def __init__(self):
        super(Local_QMG,self).__init__()
        self.CB5 = nn.Sequential(
            nn.Conv2d(in_channels= 256,out_channels= 512,kernel_size=(3,3),stride=(1,1)),
            nn.BatchNorm2d(num_features= 512),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels= 512,out_channels= 512,kernel_size=(3,3),stride=(2,2),padding=(2,2)),
            nn.BatchNorm2d(num_features= 512),
            nn.LeakyReLU()
        )
        self.CB6 = nn.Sequential(
            nn.Conv2d(in_channels= 512,out_channels= 512,kernel_size=(3,3),stride=(1,1)),
            nn.BatchNorm2d(num_features= 512),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels= 512,out_channels= 512,kernel_size=(3,3),stride=(2,2),padding=(2,2)),
            nn.BatchNorm2d(num_features= 512),
            nn.LeakyReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels= 512,out_channels= 1,kernel_size=(3,3),stride=(1,1), padding= (1,1)),
            nn.BatchNorm2d(num_features= 1),
            nn.ReLU()
        )
        self.avg_pool = nn.AvgPool2d(kernel_size=(8,8),stride=(8,8))
        self.up_sample = nn.Upsample(scale_factor= 4)
    def forward(self, data):
        (E,v) = data
        E_hat = self.avg_pool(E)
        S_L = self.CB5(v)
        S_L = self.CB6(S_L)
        S_L = self.up_sample(S_L)
        S_L = self.conv4(S_L)
        P_L = S_L * E_hat

        return P_L


class Global_QMG(nn.Module):
    def __init__(self):
        super(Global_QMG,self).__init__()
        self.conv1 = nn.Conv2d(in_channels= 256,out_channels= 32,kernel_size=(1,1),stride=(1,1))
        self.conv2 = nn.Conv2d(in_channels= 256,out_channels= 32,kernel_size=(1,1),stride=(1,1))
        self.conv3 = nn.Conv2d(in_channels= 1,out_channels= 32,kernel_size=(1,1),stride=(1,1))
        self.avg_pool = nn.AvgPool2d(kernel_size=(8,8),stride=(8,8))
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels= 32,out_channels= 1,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
            nn.BatchNorm2d(num_features= 1),
            nn.ReLU()
        )
    

    def forward(self,data):
        (E,v) = data
        H, W = E.shape[-2:]
        v1 = self.conv1(v)
        v2 = self.conv2(v)
        E_tilde = self.conv3(E)
        E_tilde = self.avg_pool(E_tilde)
        v1 = v1.flatten(start_dim= -2).permute(0,2,1)
        v2 = v2.flatten(start_dim= -2)
        E_tilde = E_tilde.flatten(start_dim= -2)
        A = torch.bmm(v1,v2)
        P_G = torch.bmm(E_tilde,A)
        P_G = P_G.view(-1,32,(int)(H/8),(int)(W/8))
        P_G = self.conv5(P_G)
        return P_G

class Regressor(nn.Module):
    def __init__(self):
        super(Regressor,self).__init__()
        self.gamma = torch.tensor(0.0,dtype=torch.float, requires_grad=True)
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=(4,4),stride=(2,2))
        self.fc = nn.Sequential(
            nn.Linear(in_features = 289,out_features = 16),
            nn.LeakyReLU(),
            nn.Dropout(p = 0.1),
            nn.Linear(in_features = 16, out_features= 1)
        )

    def forward(self,data):
        (P_L,P_G) = data
        P = P_L + self.gamma * P_G
        P = self.avg_pool(P)
        P = P.view(P.shape[0],-1)
        score = self.fc(P)

        return score



class LRDB(nn.Module):
    def __init__(self):
        super(LRDB,self).__init__()
        self.BN = Backbone_Network()
        self.QMG_L = Local_QMG()
        self.QMG_G = Global_QMG()
        self.regressor = Regressor()


    def forward(self,data):
        (I_d,E) = data
        v = self.BN(data)
        P_L = self.QMG_L((E,v))
        P_G = self.QMG_G((E,v))
        s = self.regressor((P_L,P_G))

        return s
