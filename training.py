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

class MyDataset3(Dataset):
    def __init__(self, img_path1,img_path2,img_path3, transform=None):
        super(MyDataset3, self).__init__()
        all_dir_E = list()
        for path1 in [img_path1,img_path2,img_path3]:
            print(path1)
            filelist1 = os.listdir(path1)
            for i in range(0,len(filelist1)):
                img_path = path1 + '/' + str(filelist1[i])
                img = os.listdir(img_path)
                img.sort()
                if str(filelist1[i]) == "benign":
                    for ii in range(0, len(img)):
                        pic_path_b1 = img_path + '/' + str(img[ii])
                        all_dir_E.append((pic_path_b1, 0))
                else:
                    for ii in range(0, len(img)):
                        pic_path_b1 = img_path + '/' + str(img[ii])
                        all_dir_E.append((pic_path_b1, 1))
        self.sample_list_E = all_dir_E
        self.transform = transform

    def __len__(self):
        return len(self.sample_list_E)

    def __getitem__(self, index):
        item_E1,  label = self.sample_list_E[index]
        image_E1 = cv2.imread(item_E1)
        image_E1 = Image.fromarray(image_E1)

        return image_E1,label

class MyDataset0(Dataset):
    def __init__(self, img_path1, transform=None):
        super(MyDataset0, self).__init__()
        filelist1 = os.listdir(img_path1)
        all_dir_E = list()
        for i in range(0,len(filelist1)):
            img_path = img_path1 + '/' + str(filelist1[i])
            img = os.listdir(img_path)
            img.sort()
            if str(filelist1[i]) == "benign":
                for ii in range(0, len(img)):
                    pic_path_b1 = img_path + '/' + str(img[ii])
                    all_dir_E.append((pic_path_b1, 0))
            else:
                for ii in range(0, len(img)):
                    pic_path_b1 = img_path + '/' + str(img[ii])
                    all_dir_E.append((pic_path_b1, 1))
        self.sample_list_E = all_dir_E
        self.transform = transform

    def __len__(self):
        return len(self.sample_list_E)

    def __getitem__(self, index):
        item_E1,  label = self.sample_list_E[index]
        image_E1 = cv2.imread(item_E1)
        image_E1 = Image.fromarray(image_E1)
        return image_E1,label,item_E1

class MyDataset1(Dataset):
    def __init__(self,dataset,transform1,transform2):
        self.dataset = dataset
        self.transform1 = transform1
        self.transform2 = transform2
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        img,label,name=self.dataset[idx]
        number = random.randint(0, 1)
        if number == 0:
            img1 = self.transform1(img)
            img2 = self.transform2(img)
            style1 = 0
            style2 = 1
        else:
            img1 = self.transform2(img)
            img2 = self.transform1(img)
            style1 = 1
            style2 = 0

        return img1,img2,label,style1,style2,0

class MyDataset2(Dataset):
    def __init__(self,dataset,transform1):
        self.dataset = dataset
        self.transform1 = transform1
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        img1,label,name=self.dataset[idx]
        img1 = self.transform1(img1)
        return img1,label,name

## 加载数据（Facades） ##
def loadData_train(train_dataset, batch_size, shuffle=False):
    data_transform_train = transforms.Compose([
        transforms.Resize((800, 800)),
        transforms.ColorJitter(brightness=[0, 2], contrast=[0, 2],
                               saturation=[0, 2],hue=[-0.5,0.5]),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5],),
    ])
    data_transform_train2 = transforms.Compose([
        transforms.Resize((800, 800)),
        transforms.ColorJitter(brightness=[2, 4], contrast=[2, 4],
                               saturation=[2, 4],hue=[-0.5,0.5]),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], ),
    ])
    dataset = MyDataset1(train_dataset, transform1=data_transform_train, transform2=data_transform_train2)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=32)


def loadData_val(test_dataset, batch_size, shuffle=False):
    data_transform_val = transforms.Compose([
        transforms.Resize((800,800)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    dataset = MyDataset2(test_dataset, transform1=data_transform_val)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=32)

def seed_it(seed):
    random.seed(seed) #
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) #
    torch.backends.cudnn.deterministic = True #
    torch.manual_seed(seed)
seed_it(2023)

data_dir_train_E = '/PRI1'
dataset1 = MyDataset0(data_dir_train_E)
train_size = int(len(dataset1) * 0.8)
validate_size = int(len(dataset1))-train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset1, [train_size, validate_size],generator=torch.Generator().manual_seed(6))
train_loader = loadData_train(train_dataset, 24, shuffle=True)
val_loader = loadData_val(test_dataset, 40, shuffle=True)


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

            data1 = F.softmax(output_1, dim=1)
            _, predicted = torch.max(output_1.data, 1)
            correct += predicted.eq(targets.data).cpu().sum()
            train_loss = train_loss + (celoss1+celoss2+celoss3+0.2*(mseloss1+mseloss2+celoss4)).item()
            test_data_predict1 = predicted.cpu().numpy()
            test_data_predict = np.hstack([test_data_predict, test_data_predict1])
            test_data_predict_proba.extend(data1[:, 1].detach().cpu().numpy())
            test_target1 = targets.data.cpu().numpy()
            test_target = np.hstack([test_target, test_target1])

        test_data_accuracy_score = accuracy_score(test_target, test_data_predict)
        FPR_test_data, TPR_test_data, threshold_test_data = roc_curve(test_target, test_data_predict_proba)
        test_data_roc_auc = auc(FPR_test_data, TPR_test_data)
        auc1 = test(val_loader,net, CELoss, 96)

train()
