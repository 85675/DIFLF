import torch.nn as nn
import torch
import torch.nn.functional as F

class Enc(nn.Module):
    def __init__(self, model):
        super(Enc, self).__init__()
        self.features = model
        self.max = nn.MaxPool2d((25,25), stride=(25,25))
        self.classifier = nn.Sequential(
            nn.Linear(512, 64),
            nn.BatchNorm1d(64),
        )

    def forward(self, x):
        xf1, xf2, xf3, xf4, xf5 = self.features(x)
        xl3 = self.max(xf5)
        xl3 = xl3.view(xl3.size(0), -1)
        x = self.classifier(xl3)

        return x,xf5

