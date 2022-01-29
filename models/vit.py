import torch
from vit_pytorch import ViT

"""
image_size: int.
Image size. If you have rectangular images, make sure your image size is the maximum of the width and height
patch_size: int.
Number of patches. image_size must be divisible by patch_size.
The number of patches is: n = (image_size // patch_size) ** 2 and n must be greater than 16.
num_classes: int.
Number of classes to classify.
dim: int.
Last dimension of output tensor after linear transformation nn.Linear(..., dim).
depth: int.
Number of Transformer blocks.
heads: int.
Number of heads in Multi-head Attention layer.
mlp_dim: int.
Dimension of the MLP (FeedForward) layer.
channels: int, default 3.
Number of image's channels.
dropout: float between [0, 1], default 0..
Dropout rate.
emb_dropout: float between [0, 1], default 0.
Embedding dropout rate.
pool: string, either cls token pooling or mean pooling
"""


# Input Size: 128 * 256
def vit(dataset):
    """
    Constructs a Vision Transformer
    """

    num_classes = 10 if dataset == "UrbanSound8K" else 50
    model = ViT(
        image_size=256,
        patch_size=32,
        num_classes=num_classes,
        dim=1024,
        depth=6,
        heads=16,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1
    )

    return model


# vit_model = vit("UrbanSound8K")
# img = torch.randn(1, 3, 128, 256)
# preds = vit_model(img)
#
# print(preds)
