import argparse
import os

import numpy as np
import pandas as pd

import torch
from torch.utils.data.dataloader import DataLoader

from sklearn.model_selection import train_test_split

from dataset import RsnaDataset


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
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--validation_pct", type=float, default=0.2)
    # Add new parsers here...

    args = parser.parse_args()
    return args


def get_train_val_loaders(args):
    # Load and split train and validation data
    train_data = pd.read_csv(os.path.join(args.data_path, "train_labels.csv"))

    df_train, df_validation = train_test_split(
        train_data, test_size=args.validation_pct, stratify=train_data["MGMT_value"]
    )

    train_dataset = RsnaDataset(
        paths=df_train["BraTS21ID"].values,
        targets=df_train["MGMT_value"].values,
        args=args,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    validation_dataset = RsnaDataset(
        paths=df_validation["BraTS21ID"].values,
        targets=df_validation["MGMT_value"].values,
        args=args,
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=args.batch_size, shuffle=True
    )
    return train_loader, validation_loader


if __name__ == "__main__":
    args = get_arguments()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device

    # First we work on the training data
    args.data_dir = "train"
    train_loader, validation_loader = get_train_val_loaders(args)

    # Initialize the model here...
    # model = RsnaModel()

    # Select optimizer and loss function here...
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # criterion =
