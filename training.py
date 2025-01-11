from __future__ import print_function
import os

import torch
from PIL import Image
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler
import cv2
import math
from tqdm import tqdm
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
import copy
from utilshn import *
from torch.autograd import Function


class DCNN(nn.Module):
    def __init__(self, model1,model2):
        super(DCNN, self).__init__()
        self.resnet1 = model1
        self.resnet2 = model2
        self.linear1 = torch.nn.Linear(64, 2)
        self.linear2 = torch.nn.Linear(32, 2)
        self.linear3 = torch.nn.Linear(32, 2)

    def forward(self, x1,x2):
        x11, xf51 = self.resnet1(x1)
        x22, xf52 = self.resnet2(x2)

        # divide
        style1, content1 = x11.split([32, 32], dim=1)
        style2, content2 = x22.split([32, 32], dim=1)
        content = torch.cat((content1, content2), -1)

        # FMD unit
        s51, c51 = xf51.split([256, 256], dim=1)
        s52, c52 = xf52.split([256, 256], dim=1)
        c51 = torch.nn.functional.normalize(c51.view(c51.size(0), -1), dim=1)
        s51 = torch.nn.functional.normalize(s51.view(s51.size(0), -1), dim=1)
        c52 = torch.nn.functional.normalize(c52.view(c52.size(0), -1), dim=1)
        s52 = torch.nn.functional.normalize(s52.view(s52.size(0), -1), dim=1)
        plogit = (c51 * c52).sum(dim=-1) / 0.3
        nlogit1 = (c51 * s51).sum(dim=-1) / 0.3
        nlogit2 = (c52 * s52).sum(dim=-1) / 0.3
        pologit = torch.unsqueeze(plogit, dim=1)
        nelogit1 = torch.unsqueeze(nlogit1, dim=1)
        nelogit2 = torch.unsqueeze(nlogit2, dim=1)
        vc1 = torch.cat((pologit, nelogit1, nelogit2), dim=1)

        # FVD unit
        v1 = torch.cat((content2, style1), -1)
        v2 = torch.cat((content1, style2), -1)

        return F.leaky_relu(self.linear1(content)),F.leaky_relu(self.linear2(style1)),F.leaky_relu(self.linear3(style2)),v1,v2,x11,x22,vc1



def train():
    # Model
    net1 = load_model1(model_name='resnet34', pretrain=True, require_grad=True)
    net2 = load_model2(model_name='resnet34', pretrain=True, require_grad=True)
    net = DCNN(net1,net2)
    # GPU
    device = torch.device("cuda:0")
    net.to(device)
    CELoss = nn.CrossEntropyLoss()
    criterion1 = nn.MSELoss()

    optimizer = optim.Adam([
        {'params': net.parameters(), 'lr': 1e-5},
    ],weight_decay=1e-4)
    vacc = []
    scaler1 = GradScaler()
    for epoch in range(30):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        test_target= []
        test_data_predict = []
        test_data_predict_proba = []
        for inputs1, inputs2, targets, style1,style2, zro in tqdm(train_loader):
            inputs1, inputs2, targets,style1,style2,zro = inputs1.to(device), inputs2.to(device), targets.to(device), style1.to(device),style2.to(device), zro.to(device)
            optimizer.zero_grad()
            with autocast():
                output_1,output_2,output_3,v1,v2,c1,c2, vc1 = net(inputs1,inputs2)
                # content loss
                celoss1 = CELoss(output_1, targets)

                # style loss
                celoss2 = CELoss(output_2, style1)
                celoss3 = CELoss(output_3, style2)

                # fvd loss
                mseloss1 = criterion1(v1, c1)
                mseloss2 = criterion1(v2, c2)

                # fmd loss
                celoss4 = CELoss(vc1, zro)

            scaler1.scale(celoss1+celoss2+celoss3+0.2*(mseloss1+mseloss2+celoss4)).backward()
            scaler1.step(optimizer)
            scaler1.update()

