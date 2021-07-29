import glob
import os
import random

import cv2
import numpy as np
from numpy.core.fromnumeric import sort

import torch
from torch.utils.data import Dataset, DataLoader

from utils import load_dicom


class RsnaDataset:

    """
    paths: Subject IDs from the dataset
    targets: MGMT_values for the respective subjects
    channels: The idea is to combine 4 different MRI structures as 4 channels -- FLAIR, T1w, T1wCE, and T2w
    """

    def __init__(self, paths, targets, args, transform=None) -> None:
        self.paths = paths
        self.targets = targets
        self.args = args
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx):
        sub_id = self.paths[idx]
        patient_path = os.path.join(
            self.args.data_path, self.args.data_dir, str(sub_id).zfill(5)
        )
        channels = []
        for data_type in ("FLAIR", "T1w", "T1wCE", "T2w"):
            data_path = os.path.join(patient_path, data_type)
            channels.append(self.read_images(data_path))

        X = torch.tensor(channels).float()

        return {
            "X": X.permute(1, 0, 2, 3),
            "y": torch.tensor(self.targets[idx], dtype=torch.int),
        }

    def add_img_paths(self, path):
        """
        In case when the img_directory does not contain sufficient images required to complete the
        desired sequence length, this function repeats the same earlier images by adding them at the
        end to meet the requirement
        """
        img_paths = sorted([os.path.join(path, dir) for dir in os.listdir(path)])
        new_img_paths = list(img_paths)
        while len(new_img_paths) <= self.args.sequence_length:
            remaining = self.args.sequence_length - len(new_img_paths)
            new_img_paths += img_paths[: min(remaining, len(img_paths))]

        return new_img_paths

    def read_images(self, path):
        if len(os.listdir(path)) < self.args.sequence_length:
            # In this case, we will need to restart adding from the first image to complete the sequence
            img_paths = self.add_img_paths(path)
        else:
            img_paths = random.sample(
                sorted([os.path.join(path, dir) for dir in os.listdir(path)]),
                self.args.sequence_length,
            )

        images = [
            cv2.resize(load_dicom(img_path), self.args.img_shape) / 255
            for img_path in img_paths
        ]
        return images

    # def __getitem__(self, idx):
    #     sub_id = self.paths[idx]
    #     patient_path = os.path.join(
    #         self.args.data_path, self.args.data_dir, str(sub_id).zfill(5)
    #     )

    #     channels = []
    #     for type in ("FLAIR", "T1w", "T1wCE"):  # , "T2w"
    #         # glob.glob is equivalent to os.listdir() function
    #         dcm_paths = sorted(
    #             glob.glob(os.path.join(patient_path, type, "*")),
    #             key=lambda x: int(x[:-4].split("-")[-1]),
    #         )

    #         x = len(dcm_paths)
    #         if x < 10:
    #             r = range(x)
    #         else:
    #             d = x // 10
    #             r = range(d, x - d, d)

    #         channel = []
    #         for i in r:
    #             channel.append(cv2.resize(load_dicom(dcm_paths[i]), (256, 256)) / 255)
    #             # channel.append(load_dicom(dcm_paths[i]))

    #         channel = np.mean(channel, axis=0)
    #         channels.append(channel)

    #     X = torch.tensor(channels).float()
    #     if self.transform:
    #         X = self.transform(X)

    #     y = torch.tensor(self.targets[idx], dtype=torch.float)
    #     return {"X": X, "y": y}
