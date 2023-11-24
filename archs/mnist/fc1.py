import torch
import torch.nn as nn

class fc1(nn.Module):

    def __init__(self, num_classes=10):
        super(fc1, self).__init__()
        self.fc1 = nn.Linear(28*28, 300),
        self.fc2 = nn.Linear(300, 100),
        self.fc3 = nn.Linear(100, num_classes),
        self.activation = nn.ReLU(inplace=True),
        

    def forward(self, x):
        x = torch.flatten(x, 1)
        h1 = self.activation(self.fc1(x))
        h2 = self.activation(self.fc2(h1))
        h3 = self.activation(self.fc3(h2))
        return h1, h2, h3

    