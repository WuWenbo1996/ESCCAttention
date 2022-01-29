"""
See flops and params of the model
"""
from thop import profile
from thop import clever_format
import torch
import models.resnet


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = models.resnet.resnet50(dataset='ESC', pretrained=True).to(device)
input = torch.randn(1, 3, 128, 250)
flops, params = profile(model, inputs=(input, ))
flops, params = clever_format([flops, params], "%.3f")
print(flops, params)