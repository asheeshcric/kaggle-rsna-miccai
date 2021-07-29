import argparse
import numpy as np
import pydicom


def get_arguments():
    parser = argparse.ArgumentParser(
        description="Kaggle RSNA Brain Tumor Detection Challenge Code"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/home/asheesh/Documents/Github/kaggle-rsna-miccai/data/rsna-miccai-brain-tumor-radiogenomic-classification",
        help="Path to the main dataset directory that contains both train and test data",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--img_shape", type=int, default=256)
    parser.add_argument("--validation_pct", type=float, default=0.2)
    parser.add_argument("--input_channels", type=int, default=4)
    parser.add_argument("--output_channels", type=int, default=64)
    parser.add_argument("--lstm_hidden_size", type=int, default=128)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=30,
        help="Lenght of image sequence that you want for the model",
    )
    # Add new parsers here...

    args = parser.parse_args()
    return args


def load_dicom(path):
    dicom = pydicom.read_file(path)
    data = dicom.pixel_array
    data = data - np.min(data)
    if np.max(data) != 0:
        data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    return data


class LossMeter:
    def __init__(self) -> None:
        self.avg = 0
        self.n = 0

    def update(self, val):
        self.n += 1
        self.avg = val / self.n + (self.n - 1) / self.n * self.avg


class AccMeter:
    def __init__(self) -> None:
        self.avg = 0
        self.n = 0

    def update(self, y_true, y_pred):
        y_true = y_true.cpu().numpy().astype(int)
        y_pred = y_pred.cpu().numpy() >= 0
        last_n = self.n
        self.n += len(y_true)
        true_count = np.sum(y_true == y_pred)
        self.avg = true_count / self.n + last_n / self.n * self.avg
