import argparse
import os
import time

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader

from sklearn.model_selection import train_test_split

from dataset import RsnaDataset

from utils import LossMeter, AccMeter, get_arguments
from model import RsnaModel

"""
python main.py --epochs=5 --batch_size=4 --validation_pct=0.2

"""


def get_train_val_loaders(args, transform=None):
    # Load and split train and validation data
    train_data = pd.read_csv(os.path.join(args.data_path, "train_labels.csv"))

    df_train, df_validation = train_test_split(
        train_data, test_size=args.validation_pct, stratify=train_data["MGMT_value"]
    )

    train_dataset = RsnaDataset(
        paths=df_train["BraTS21ID"].values,
        targets=df_train["MGMT_value"].values,
        args=args,
        transform=transform,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    validation_dataset = RsnaDataset(
        paths=df_validation["BraTS21ID"].values,
        targets=df_validation["MGMT_value"].values,
        args=args,
        transform=transform,
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=args.batch_size, shuffle=True
    )
    return train_loader, validation_loader


class Trainer:
    def __init__(
        self, args, model, optimizer, criterion, loss_meter, score_meter
    ) -> None:
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.loss_meter = loss_meter
        self.score_meter = score_meter

        self.best_valid_score = -np.inf
        self.n_patience = 0

        self.messages = {
            "epoch": "[Epoch {}: {}] loss: {:.5f}, score: {:.5f}, time: {} s",
            "checkpoint": "The score improved from {:.5f} to {:.5f}. Save model to '{}'",
            "patience": "\nValid score didn't improve last {} epochs.",
        }

    def fit(self, train_loader, validation_loader, save_path, patience):
        for n_epoch in range(1, self.args.epochs + 1):
            self.info_message(f"EPOCH: {n_epoch}")

            train_loss, train_score, train_time = self.train_epoch(train_loader)
            validation_loss, validation_score, validation_time = self.valid_epoch(
                validation_loader
            )

            self.info_message(
                self.messages["epoch"],
                "Train",
                n_epoch,
                train_loss,
                train_score,
                train_time,
            )

            self.info_message(
                self.messages["epoch"],
                "Valid",
                n_epoch,
                validation_loss,
                validation_score,
                validation_time,
            )

            if self.best_valid_score < validation_score:
                self.info_message(
                    self.messages["checkpoint"],
                    self.best_valid_score,
                    validation_score,
                    save_path,
                )
                self.best_valid_score = validation_score
                self.save_model(n_epoch, save_path)
                self.n_patience = 0
            else:
                self.n_patience += 1

            if self.n_patience >= patience:
                self.info_message(self.messages["patience"], patience)
                break

    def train_epoch(self, train_loader):
        self.model.train()
        t = time.time()
        train_loss = self.loss_meter()
        train_score = self.score_meter()

        for step, batch in enumerate(train_loader, 1):
            X = batch["X"].to(self.args.device)
            targets = batch["y"].to(self.args.device)
            self.optimizer.zero_grad()
            outputs = self.model(X).squeeze(1)

            loss = self.criterion(outputs, targets)
            loss.backward()

            train_loss.update(loss.detach().item())
            train_score.update(targets, outputs.detach())

            self.optimizer.step()

            _loss, _score = train_loss.avg, train_score.avg
            message = "Train Step {}/{}, train_loss: {:.5f}, train_score: {:.5f}"
            self.info_message(message, step, len(train_loader), _loss, _score, end="\r")

        return train_loss.avg, train_score.avg, int(time.time() - t)

    def valid_epoch(self, valid_loader):
        self.model.eval()
        t = time.time()
        valid_loss = self.loss_meter()
        valid_score = self.score_meter()

        for step, batch in enumerate(valid_loader, 1):
            with torch.no_grad():
                X = batch["X"].to(self.args.device)
                targets = batch["y"].to(self.args.device)

                outputs = self.model(X).squeeze(1)
                loss = self.criterion(outputs, targets)

                valid_loss.update(loss.detach().item())
                valid_score.update(targets, outputs)

            _loss, _score = valid_loss.avg, valid_score.avg
            message = "Valid Step {}/{}, valid_loss: {:.5f}, valid_score: {:.5f}"
            self.info_message(message, step, len(valid_loader), _loss, _score, end="\r")

        return valid_loss.avg, valid_score.avg, int(time.time() - t)

    def save_model(self, n_epoch, save_path):
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_valid_score": self.best_valid_score,
                "n_epoch": n_epoch,
            },
            save_path,
        )

    @staticmethod
    def info_message(message, *args, end="\n"):
        print(message.format(*args), end=end)


if __name__ == "__main__":
    args = get_arguments()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device
    args.img_shape = (args.img_shape, args.img_shape)

    # First we work on the training data
    args.data_dir = "train"
    train_loader, validation_loader = get_train_val_loaders(args)

    # Initialize the model here...
    model = RsnaModel()
    model.to(args.device)

    # Select optimizer and loss function here...
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = F.binary_cross_entropy_with_logits

    # Start training
    trainer = Trainer(args, model, optimizer, criterion, LossMeter, AccMeter)

    history = trainer.fit(train_loader, validation_loader, f"best_model_0.pth", 100)
