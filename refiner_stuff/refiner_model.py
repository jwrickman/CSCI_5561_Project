import torch as pt
from torch.nn import Conv2d, ConvTranspose2d, Module, ReLU
import torch.nn as nn
import torch.nn.functional as F


class RefinerModel(Module):
    def __init__(self):
        super(RefinerModel, self).__init__()
        self.conv1 = Conv2d(3, 32, kernel_size=(5,5))
        self.conv2 = Conv2d(32, 64, kernel_size=(3,3))
        self.conv3 = Conv2d(64, 128, kernel_size=(3,3))
        self.convt1 = ConvTranspose2d(128, 64, kernel_size=(3,3))
        self.convt2 = ConvTranspose2d(64, 32, kernel_size=(3,3))
        self.convt3 = ConvTranspose2d(32, 16, kernel_size=(5,5))
        self.output = Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.convt1(x))
        x = F.interpolate(x, scale_factor=2)
        x = F.relu(self.convt2(x))
        x = F.interpolate(x, scale_factor=2)
        x = F.relu(self.convt3(x))
        x = self.output(x)
        return x



class RefinerModelRegression(Module):
    def __init__(self):
        super(RefinerModelRegression, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.fc1 = nn.Linear(128, 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(p=0.2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        bs, _, _, _ = x.shape
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        x = self.dropout(x)
        out = self.fc1(x)
        return out
