import glob
import os

import cv2
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

from utils import load_dicom


class RsnaDataset:

    """
    paths: Subject IDs from the dataset
    targets: MGMT_values for the respective subjects
    channels: The idea is to combine 4 different MRI structures as 4 channels -- FLAIR, T1w, T1wCE, and T2w
    """

    def __init__(self, paths, targets, args) -> None:
        self.paths = paths
        self.targets = targets
        self.args = args

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx):
        sub_id = self.paths[idx]
        patient_path = os.path.join(
            self.args.data_path, self.args.data_dir, str(sub_id).zfill(5)
        )

        channels = []
        for type in ("FLAIR", "T1w", "T1wCE"): # , "T2w"
            # glob.glob is equivalent to os.listdir() function
            dcm_paths = sorted(
                glob.glob(os.path.join(patient_path, type, "*")),
                key=lambda x: int(x[:-4].split("-")[-1]),
            )

            x = len(dcm_paths)
            if x < 10:
                r = range(x)
            else:
                d = x // 10
                r = range(d, x - d, d)

            channel = []
            for i in r:
                channel.append(cv2.resize(load_dicom(dcm_paths[i]), (256, 256)) / 255)

            channel = np.mean(channel, axis=0)
            channels.append(channel)

        y = torch.tensor(self.targets[idx], dtype=torch.float)
        return {"X": torch.tensor(channels).float(), "y": y}
