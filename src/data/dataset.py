import os
from collections import defaultdict

import cv2
import numpy as np
import torch


def normalize_input(image):
    return image / 127.5 - 1.0


def compute_data_mean(files):
    total = np.zeros(3)
    for imgPath in files:
        img = cv2.imread(imgPath)
        total += img.mean(axis=(0, 1))
    channel_mean = total / len(files)
    mean = np.mean(channel_mean)

    return mean - channel_mean[..., ::-1]  # Convert to BGR for training


class AnimeDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, style: str, isTrain=True, transform=None):
        self.image_files = {}
        self.style = f"{style}/style"
        self.smooth = f"{style}/smooth"
        if isTrain is True:
            self.photo = "train_photo"
        else:
            self.photo = "val"
            # self.transform
        for photo_type in [self.photo, self.style, self.smooth]:
            folder = os.path.join(data_path, photo_type)
            files = os.listdir(folder)
            self.image_files[photo_type] = [
                os.path.join(folder, filePath) for filePath in files
            ]

        self.transform = transform
        self.mean = compute_data_mean(self.image_files[self.style])

    @property
    def len_anime(self):
        return len(self.image_files[self.style])

    @property
    def len_smooth(self):
        return len(self.image_files[self.smooth])

    def __len__(self):
        return len(self.image_files[self.photo])

    def __getitem__(self, index):
        image = self.load_photo(index)
        anm_idx = index
        if anm_idx > self.len_anime - 1:
            anm_idx -= self.len_anime * (index // self.len_anime)

        anime, anime_gray = self.load_anime(anm_idx)
        smooth_gray = self.load_anime_smooth(anm_idx)

        return (
            image.type(torch.float),
            anime.type(torch.float),
            anime_gray.type(torch.float),
            smooth_gray.type(torch.float),
        )

    def load_photo(self, index):
        fpath = self.image_files[self.photo][index]
        image = cv2.imread(fpath)[:, :, ::-1]
        image = self._transform(image, addmean=False)
        if self.photo == "val":
            image = cv2.resize(image, (256, 256))
        image = image.transpose(2, 0, 1)
        return torch.tensor(image)

    def load_anime(self, index):
        fpath = self.image_files[self.style][index]
        image = cv2.imread(fpath)[:, :, ::-1]

        image_gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
        image_gray = np.stack([image_gray, image_gray, image_gray], axis=-1)
        image_gray = self._transform(image_gray, addmean=False)
        image_gray = image_gray.transpose(2, 0, 1)

        image = self._transform(image, addmean=True)
        image = image.transpose(2, 0, 1)
        return torch.tensor(image), torch.tensor(image_gray)

    def load_anime_smooth(self, index):
        fpath = self.image_files[self.smooth][index]
        image = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        image = np.stack([image, image, image], axis=-1)
        image = self._transform(image, addmean=False)
        image = image.transpose(2, 0, 1)
        return torch.tensor(image)

    def _transform(self, image, addmean=False):
        if self.transform is not None:
            image = self.transform(image=image)["image"]

        imag = image.astype(np.float32)
        if addmean:
            image = image + self.mean

        return normalize_input(image)
