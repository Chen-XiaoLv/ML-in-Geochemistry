import torch
import torch.nn as nn

class Net1(nn.Module):

    def __init__(self):
        super(Net1, self).__init__()
        self.hidden=nn.Sequential(
            nn.Linear(54,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,7)
        )

    def forward(self,x):
        return nn.functional.softmax(self.hidden(x))
