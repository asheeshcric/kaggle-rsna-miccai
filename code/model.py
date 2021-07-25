import torch
import torch.nn as nn

from efficientnet_pytorch import EfficientNet


class RsnaModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = EfficientNet.from_pretrained("efficientnet-b0")
        checkpoint = torch.load("/home/asheesh/Documents/Github/kaggle-rsna-miccai/checkpoints/efficientnet-b0-08094119.pth")
        self.net.load_state_dict(checkpoint)
        n_features = self.net._fc.in_features
        self.net._fc = nn.Linear(in_features=n_features, out_features=1, bias=True)

    def forward(self, x):
        out = self.net(x)
        return out
