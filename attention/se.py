import torch.nn as nn


class se_layer(nn.Module):
    """Construct a Channel Squeeze-Excitation Module
    """

    def __init__(self, channel, reduction=16):
        super(se_layer, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)

        self.excitation = nn.Sequential(
            nn.Linear(channel, channel//reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel//reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.shape

        # Squeeze
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)

        # Excitation
        return x * y.expand_as(x)


