import torch
import torch.nn as nn

from efficientnet_pytorch import EfficientNet


class RsnaModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = EfficientNet.from_pretrained('efficientnet-b0')
