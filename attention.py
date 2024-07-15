import torch
import torch.nn as nn
import math

def conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1, bias=bias)

class CA(nn.Module):
    def __init__(self, head_dim=32):
        super(CA, self).__init__()

        self.k = nn.Sequential(*[conv(head_dim, head_dim, 3), nn.InstanceNorm3d(head_dim), nn.ReLU(), conv(head_dim, head_dim, 3)])
        self.q = nn.Sequential(*[conv(head_dim, head_dim, 3), nn.InstanceNorm3d(head_dim), nn.ReLU(), conv(head_dim, head_dim, 3), nn.Sigmoid()])
        self.v = nn.Sequential(*[conv(head_dim, head_dim, 3), nn.InstanceNorm3d(head_dim), nn.ReLU(), conv(head_dim, head_dim, 3)])
        # self.gamma = nn.Parameter(torch.zeros(1))

        # self.q = nn.Linear(head_dim, head_dim, bias=False)
        # self.k = nn.Linear(head_dim, head_dim, bias=False)
        # self.v = nn.Linear(head_dim, head_dim, bias=False)
        self.dk = math.sqrt(head_dim)

    def forward(self, x, y):
        query = self.q(y)
        key = self.k(x)
        value = self.v(x)

        energy = query * key / self.dk

        attention = torch.softmax(energy, dim = -1)

        # weight = self.k(x) * self.q(y)
        output = x + attention * value
        return output
    
class SA(nn.Module):
    def __init__(self, head_dim=32):
        super(SA, self).__init__()

        self.k = nn.Sequential(*[conv(head_dim, head_dim, 3), nn.InstanceNorm3d(head_dim), nn.ReLU(), conv(head_dim, head_dim, 3)])
        self.q = nn.Sequential(*[conv(head_dim, head_dim, 3), nn.InstanceNorm3d(head_dim), nn.ReLU(), conv(head_dim, head_dim, 3), nn.Sigmoid()])
        self.v = nn.Sequential(*[conv(head_dim, head_dim, 3), nn.InstanceNorm3d(head_dim), nn.ReLU(), conv(head_dim, head_dim, 3)])
        # self.gamma = nn.Parameter(torch.zeros(1))

        # self.q = nn.Linear(head_dim, head_dim, bias=False)
        # self.k = nn.Linear(head_dim, head_dim, bias=False)
        # self.v = nn.Linear(head_dim, head_dim, bias=False)
        self.dk = math.sqrt(head_dim)

    def forward(self, x):
        query = self.q(x)
        key = self.k(x)
        value = self.v(x)

        energy = query * key / self.dk

        attention = torch.softmax(energy, dim = -1)

        # weight = self.k(x) * self.q(y)
        output = x + attention * value
        return output
        
class SpatialAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, o, p):
        p_conv1 = self.conv1(p)
        p_att = torch.sigmoid(p_conv1)

        # 在原始特征图乘以注意力机制
        o_weighted = o * p_att
        out = self.gamma * (self.conv2(o_weighted)) + o

        return out