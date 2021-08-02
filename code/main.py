import argparse
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from dataset import RsnaDataset

from utils import LossMeter, AccMeter, get_arguments
from model import RsnaCustomNet

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


def get_confusion_matrix(args, preds, actual):
    preds = [int(k) for k in preds]
    actual = [int(k) for k in actual]

    cf = confusion_matrix(actual, preds, labels=list(range(args.num_classes)))
    return cf


def test(model, data_loader, args):
    correct, total = 0, 0
    preds, actual = [], []
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            if not batch:
                continue
            inputs, labels = batch[0].to(args.device), batch[1].to(args.device)
            outputs = model(inputs)
            _, class_pred = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (class_pred == labels).sum().item()
            preds.extend(list(class_pred.to(dtype=torch.int64)))
            actual.extend(list(labels.to(dtype=torch.int64)))

    acc = 100 * (correct / total)
    model.train()
    return preds, actual, acc


def train(model, train_loader, val_loader, args, optimizer, criterion):
    # loss_function = torch.nn.CrossEntropyLoss(weight=args.class_weights)
    # optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Star the training
    print(f"Training...")
    for epoch in range(args.epochs):
        for batch in train_loader:
            inputs, labels = batch[0].to(args.device), batch[1].to(args.device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            # Clip the gradients because I'm using LSTM layer in the model
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()

        if epoch % 2 != 0:
            # Check train and val accuracy after every two epochs
            print("Validating...")
            _, _, train_acc = test(model, train_loader, args)
            _, _, val_acc = test(model, val_loader, args)
            print(
                f"Epoch: {epoch+1} | Loss: {loss} | Train Acc: {train_acc} | Validation Acc: {val_acc}"
            )
        else:
            print("Training epoch...")
            print(f"Epoch: {epoch+1} | Loss: {loss}")

        # Save checkpoint after every 10 epochs
        if (epoch + 1) % 10 == 0:
            current_time = datetime.now().strftime("%m_%d_%Y_%H_%M")
            torch.save(
                model.state_dict(),
                f"{args.file_name}-{current_time}-lr-{args.learning_rate}-epochs-{epoch+1}-acc-{val_acc:.2f}.pth",
            )

    print("Training complete")
    return model


if __name__ == "__main__":
    args = get_arguments()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device
    args.img_shape = (args.img_shape, args.img_shape)

    # First we work on the training data
    args.data_dir = "train"
    train_loader, validation_loader = get_train_val_loaders(args)

    # temporary testing
    # data = next(iter(train_loader))
    # x, y = data['X'], data['y']
    # print(x.shape)

    # Initialize the model here...
    model = RsnaCustomNet(args)

    # DataParallel Settings
    args.num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {args.num_gpus}")
    if args.device.type == "cuda" and args.num_gpus > 1:
        model = torch.nn.DataParallel(model, list(range(args.num_gpus)))
    else:
        model.to(args.device)

    # Select optimizer and loss function here...
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = F.binary_cross_entropy_with_logits

    # Start training
    # trainer = Trainer(args, model, optimizer, criterion, LossMeter, AccMeter)

    # history = trainer.fit(train_loader, validation_loader, f"best_model_0.pth", 100)

    # My method
    model = train(model, train_loader, validation_loader, args, optimizer, criterion)

    # Validate the model
    preds, actual, acc = test(model, validation_loader)
    print(f"Validation accuracy: {acc}")
    print(get_confusion_matrix(args, preds, actual))
