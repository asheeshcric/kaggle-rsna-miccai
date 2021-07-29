import torch
import torch.nn as nn

from efficientnet_pytorch import EfficientNet


class RsnaCustomNet(nn.Module):
    """
    This model is designed to handle temporal image data in the following shape:
    (30, 4, 256, 256)
    """

    def __init__(self, args):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(
                args.input_channels,
                args.output_channels,
                kernel_size=3,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm2d(args.output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                args.output_channels,
                args.output_channels * 2,
                kernel_size=3,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm2d(args.output_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                args.output_channels,
                args.output_channels * 4,
                kernel_size=3,
                stride=1,
                bias=False,
            ),
            nn.BatchNorm2d(args.output_channels * 4),
        )

        self._to_linear, self._to_lstm = None, None
        # testing output shape with a random batch
        x = torch.randn(
            args.batch_size,
            args.sequence_length,
            args.input_channels,
            args.img_shape[0],
            args.img_shape[1],
        )
        # An initial pass to find the output shape
        self.convs(x)

        # LSTM layer for the sequential input
        self.lstm = nn.LSTM(
            input_size=self._to_lstm,
            hidden_size=args.lstm_hidden_size,
            num_layers=1,
            batch_first=True,
        )

        # Final linear layer for binary classification
        self.fc = nn.Linear(args.lstm_hidden_size, args.num_classes)

    def convs(self, x):
        batch_size, timesteps, c, h, w = x.size()
        x = x.view(batch_size * timesteps, c, h, w)
        x = self.conv(x)

        if self._to_linear is None:
            # Just runs during model initialization to calculate output size of conv layer
            self._to_linear = int(x[0].shape[0] * x[0].shape[1] * x[0].shape[2])
            r_in = x.view(batch_size, timesteps, -1)
            self._to_lstm = r_in.shape[2]

        return x

    def forward(self, x):
        batch_size, timesteps, c, h, w = x.size()
        cnn_out = self.convs(x)

        # Prepare the output from CNN to pass through the LSTM layer
        r_in = cnn_out.view(batch_size, timesteps, -1)

        # Flattening is required when we use DataParallel
        self.lstm.flatten_parameters()

        # Get output from the LSTM
        r_out, (h_n, h_c) = self.lstm(r_in)

        # Pass the output of the LSTM to FC layers
        r_out = self.fc(r_out[:, -1, :])

        # Apply softmax to the output and return it
        return r_out


class RsnaEfficientNet(nn.Module):
    # Works for 2D inputs of (3, 256, 256)
    def __init__(self):
        super().__init__()

        self.net = EfficientNet.from_pretrained("efficientnet-b0")
        checkpoint = torch.load(
            "/home/asheesh/Documents/Github/kaggle-rsna-miccai/checkpoints/efficientnet-b0-08094119.pth"
        )
        self.net.load_state_dict(checkpoint)
        n_features = self.net._fc.in_features
        self.net._fc = nn.Linear(in_features=n_features, out_features=1, bias=True)

    def forward(self, x):
        out = self.net(x)
        return out
