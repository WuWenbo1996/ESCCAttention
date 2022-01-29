"""
Check for you CUDA running environment
"""
import torch

print(torch.__version__)               # PyTorch version
print(torch.version.cuda)              # Corresponding CUDA version
print(torch.backends.cudnn.version())  # Corresponding cuDNN version
print(torch.cuda.get_device_name(0))   # GPU type
print(torch.cuda.is_available())       # Check whether GPU support