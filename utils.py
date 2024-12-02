import numpy as np
import random
import pandas as pd
import torch
from torchvision import datasets, models, transforms
import torchvision
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score, roc_curve, recall_score, accuracy_score, auc, f1_score, \
    classification_report, precision_score, confusion_matrix
import torch.nn.functional as F
from Resnet import *
from sklearn.metrics import roc_auc_score, roc_curve, recall_score, accuracy_score, auc
import matplotlib.pyplot as plt


def load_model1(model_name, pretrain=True, require_grad=True):
    print('==> Building model..')
    if model_name == 'resnet34':
        net1 = resnet34(pretrained=True)
        for param in net1.parameters():
            param.requires_grad = require_grad
        net1 = Enc(net1)
    return net1

def load_model2(model_name, pretrain=True, require_grad=True):
    print('==> Building model..')
    if model_name == 'resnet34':
        net2 = resnet34s(pretrained=True)
        for param in net2.parameters():
            param.requires_grad = require_grad
        net2 = Enc(net2)
    return net2


def test(test_loader,net,criterion, batch_size):
    net.eval()
    use_cuda = torch.cuda.is_available()
    correct = 0
    total = 0
    test_target = []
    test_data_predict = []
    test_data_predict_proba = []
    device = torch.device("cuda:0")
    with torch.no_grad():
        for batch_idx, (inputs1,  targets, name) in enumerate(test_loader):
            if use_cuda:
                inputs1, targets = inputs1.to(device), targets.to(device)
            output_concat,output_2,output_3,v1,v2,c1,c2,vc1 = net(inputs1,inputs1)
            data1 = F.softmax(output_concat, dim=1)
            test_data_predict_proba.extend(data1[:, 1].detach().cpu().numpy())

            _, predicted = torch.max(output_concat.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            test_target1 = targets.data.cpu().numpy()
            test_target = np.hstack([test_target, test_target1])
            test_data_predict1 = predicted.cpu().numpy()
            test_data_predict = np.hstack([test_data_predict, test_data_predict1])

        FPR_test_data, TPR_test_data, threshold_test_data = roc_curve(test_target, test_data_predict_proba)
        test_data_roc_auc = auc(FPR_test_data, TPR_test_data)

        return test_data_roc_auc


