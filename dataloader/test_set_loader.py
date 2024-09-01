from os import path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder


class ImageList(ImageFolder):
    def __init__(self, source, image_list, rgb):
        self.samples = np.asarray(pd.read_csv(image_list, header=None)).squeeze(1)
        # sec = 1
        # chunk = self.samples.shape[0] // 4
        # self.samples = self.samples[sec * chunk: (sec + 1) * chunk]

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((112, 112), antialias=True),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self.rgb = rgb

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        # checks if input is RGB or BGR
        if self.rgb:
            img = Image.open(self.samples[index])
        else:
            img = cv2.imread(self.samples[index])
        return self.transform(img), self.samples[index]


class TestDataLoader(DataLoader):
    def __init__(
            self, batch_size, workers, source, image_list, rgb=True
    ):
        self._dataset = ImageList(source, image_list, rgb)

        super(TestDataLoader, self).__init__(
            self._dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=workers,
            drop_last=False,
        )

