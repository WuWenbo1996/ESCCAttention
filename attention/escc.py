import torch.nn as nn
import torch


class my_layer(nn.Module):
    """
    Construct paper attention model
    """
    def __init__(self, channel, reduction=16):
        super(my_layer, self).__init__()
        # Global
        self.f_global = nn.AdaptiveAvgPool2d(1)

        # Local
        self.t_avg_pool = nn.AdaptiveAvgPool2d((1, None))
        self.s_avg_pool = nn.AdaptiveAvgPool2d((None, 1))

        self.t_conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(1)
        )
        self.s_conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(1)
        )

        # Fusion
        self.ts_fusion = nn.Linear(in_features=3, out_features=1)

        # Channel Attention
        self.channel_weight = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.shape

        f_g = self.f_global(x).view(b, c, 1)

        out = x.reshape(b * c, 1, h, w)

        # temporal attention
        xt = self.t_avg_pool(out).view(b * c, 1, w)
        xt = self.t_conv(xt).view(b, c, 1)

        # spectral attention
        xs = self.s_avg_pool(out).view(b * c, 1, h)
        xs = self.s_conv(xs).view(b, c, 1)

        fusion_l = torch.cat([xt, xs], dim=2)
        fusion_gl = torch.cat([f_g, fusion_l], dim=2)

        c_feature = self.ts_fusion(fusion_gl).view(b, c)

        c_weight = self.channel_weight(c_feature).view(b, c, 1, 1)

        return x * c_weight.expand_as(x)