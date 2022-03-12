import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, num_features=96, num_classes=16):
        super(Net, self).__init__()
        self.fc0 = nn.Linear(num_features, 512)
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.bn1 = nn.BatchNorm1d(512, momentum=0.05)
        
    def forward(self, x):
        x = self.fc0(x)
        x = F.leaky_relu(x,0.05)
        x = self.bn1(x)
        x = self.fc1(x)
        x = F.leaky_relu(x,0.05)
        x = self.fc2(x)
        return x
