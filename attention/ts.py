import torch
import torch.nn as nn


class TemporalAttention(nn.Module):
    def __init__(self, channel):
        super(TemporalAttention, self).__init__()

        self.global_temporal = nn.Conv2d(in_channels=channel, out_channels=1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.global_temporal(x)
        out = torch.mean(out, dim=3, keepdim=True)
        out = self.sigmoid(out)
        return x * out.expand_as(x)


class SpectralAttention(nn.Module):
    def __init__(self, channel):
        super(SpectralAttention, self).__init__()

        self.global_spectral = nn.Conv2d(in_channels=channel, out_channels=1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.global_spectral(x)
        out = torch.mean(out, dim=2, keepdim=True)
        out = self.sigmoid(out)
        return x * out.expand_as(x)


class ts_layer(nn.Module):
    def __init__(self, channel):
        super(ts_layer, self).__init__()

        self.ta = TemporalAttention(channel)
        self.sa = SpectralAttention(channel)

        self.weight = nn.Parameter(torch.tensor([0.33, 0.33, 0.33]), requires_grad=True)

    def forward(self, x):
        b, c, h, w = x.size()

        # Channel attention Module
        Ut = self.ta(x)
        # Spatial attention Module
        Us = self.sa(x)

        weight = F.softmax(self.weight, dim=0)

        out = weight[0] * Ut + weight[1] * x + weight[2] * Us
        return out