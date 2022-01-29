'''
Author: wuwenbo
Date: 2021-06-02 10:14:54
LastEditTime: 2021-06-19 16:42:03
FilePath: /ESCCAttention/dataloaders.py
'''
import torch
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

class FrequencyMask(object):
    """
    Implements frequency masking transform from SpecAugment paper (https://arxiv.org/abs/1904.08779)

    Example:
    >>> transforms.Compose([
    >>>     transforms.ToTensor(),
    >>>     FrequencyMask(max_width=10, use_mean=False),
    >>> ])
    """

    def __init__(self, max_width, use_mean=True):
        self.max_width = max_width
        self.use_mean = use_mean

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) where the frequency mask is to be applied.
        Returns:
            Tensor: Transformed image with Frequency Mask.
        """
        start = random.randrange(0, tensor.shape[0])
        end = start + random.randrange(0, self.max_width)
        if self.use_mean:
            tensor[start:end, :, :] = tensor.mean()
        else:
            tensor[start:end, :, :] = 0
        return tensor

    def __repr__(self):
        format_string = self.__class__.__name__ + "(max_width="
        format_string += str(self.max_width) + ")"
        return format_string


class TimeMask(object):
    """
    Implements time masking transform from SpecAugment paper (https://arxiv.org/abs/1904.08779)

    Example:
    >>> transforms.Compose([
    >>>     transforms.ToTensor(),
    >>>     TimeMask(max_width=10, use_mean=False),
    >>> ])
    """

    def __init__(self, max_width, use_mean=True):
        self.max_width = max_width
        self.use_mean = use_mean

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) where the time mask is to be applied.
        Returns:
            Tensor: Transformed image with Time Mask.
        """
        start = random.randrange(0, tensor.shape[1])
        end = start + random.randrange(0, self.max_width)
        if self.use_mean:
            tensor[:, start:end, :] = tensor.mean()
        else:
            tensor[:, start:end, :] = 0
        return tensor

    def __repr__(self):
        format_string = self.__class__.__name__ + "(max_width="
        format_string += str(self.max_width) + ")"
        return format_string


class AudioDataset(Dataset):
    def __init__(self, audio_df, transforms=None):
        self.audio_df = audio_df
        self.transforms = transforms

    def __len__(self):
        return len(self.audio_df)

    def __getitem__(self, index):
        img_file_name, label, fold = self.audio_df.iloc[index]
        spectrogram = np.array(plt.imread(img_file_name))
        # print(spectrogram)

        if self.transforms is not None:
            spectrogram = self.transforms(spectrogram)

        return {'spectrogram': spectrogram.float(), 'label': label}
