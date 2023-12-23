import torch.nn as nn
import torch

def lrelu(x, leak=0.2):
    return torch.maximum(x, x * leak)

class own_relu(nn.Module):
    def __init__(self):
        super(own_relu, self).__init__()

    def forward(self, x):
        x = lrelu(x)
        return x

class Reshape(nn.Module):
    def __init__(self,outshape):
        super(Reshape, self).__init__()
        self.outshape=outshape

    def forward(self,x):
        return x.view(self.outshape[0],self.outshape[1],self.outshape[2],self.outshape[3])
