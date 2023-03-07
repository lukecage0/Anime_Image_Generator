import dataclasses
import os

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torch.optim as optim
from omegaconf import OmegaConf

from data.dataset import AnimeDataset
from loss_function.loss import AnimeGANLoss
from modelling.discriminator import Discriminator
from modelling.generator import Generator
from modelling.vgg_features import Vgg19


def denormalize_input(images, dtype=None):
    """
    [-1, 1] -> [0, 255]
    """
    images = images * 127.5 + 127.5

    if dtype is not None:
        if isinstance(images, torch.Tensor):
            images = titleimages.type(dtype)
        else:
            # numpy.ndarray
            images = images.astype(dtype)

    return images


def save_checkpoint(model, epoch, title):
    checkpoint = {"model_state_dict": model.state_dict(), "epoch": epoch}
    data_dir = "./saved_models"
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    torch.save(checkpoint, f"{data_dir}/{title}_{epoch}.pt")


class AnimeGANTrainer(pl.LightningModule):
    def __init__(self, params_path):
        super().__init__()
        self.params = OmegaConf.load(params_path)
        self.generator = Generator()
        self.discriminator = Discriminator(params=self.params)
        self.vgg = Vgg19()
        self.loss_fn = AnimeGANLoss(params=self.params)

        self._rgb_to_yuv_kernel = (
            torch.tensor(
                [
                    [0.299, -0.14714119, 0.61497538],
                    [0.587, -0.28886916, -0.51496512],
                    [0.114, 0.43601035, -0.10001026],
                ]
            )
            .float()
            .to("cuda")
        )
        self.loss_fn = AnimeGANLoss(params=self.params)

    def rgb_to_yuv(self, image):
        """
        https://en.wikipedia.org/wiki/YUV
        output: Image of shape (H, W, C) (channel last)
        """
        # -1 1 -> 0 1
        image = (image + 1.0) / 2.0

        yuv_img = torch.tensordot(
            image, self._rgb_to_yuv_kernel, dims=([image.ndim - 3], [0])
        )

        return yuv_img

    def content_step(self, batch):
        image, *_ = batch
        fake_image = self.generator(image)
        loss = self.loss_fn.vgg_content_loss(self.vgg, image, fake_image)
        return loss

    def disc_step(self, batch):
        image, anime_image, anime_gray, anime_smooth_gray = batch
        fake_image = self.generator(image).detach()
        fake_d = self.discriminator(fake_image)
        real_anime_d = self.discriminator(anime_image)
        real_anime_gray_d = self.discriminator(anime_gray)
        real_anime_smooth_d = self.discriminator(anime_smooth_gray)

        loss_d = self.loss_fn.compute_discriminator_loss(
            fake_d, real_anime_d, real_anime_gray_d, real_anime_smooth_d
        )
        return loss_d

    def gen_step(self, batch):
        image, anime_image, anime_gray, anime_smooth_gray = batch
        fake_image = self.generator(image)
        fake_d = self.discriminator(fake_image)

        adv_loss, con_loss, gra_loss, col_loss = self.loss_fn.compute_generator_loss(
            self.vgg, self.rgb_to_yuv, fake_image, image, fake_d, anime_gray
        )

        loss_g = adv_loss + con_loss + gra_loss + col_loss
        return loss_g

    def on_train_start(self):
        self.logger.log_hyperparams(self.params)

    def training_step(self, batch, batch_idx, optimizer_idx):

        if self.current_epoch < self.params.init_epoch:
            # print("Here")
            if optimizer_idx == 0:
                content_loss = self.content_step(batch)
                self.log(
                    "content_loss",
                    content_loss,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                )
                self.logger.log_metrics({"content_loss": content_loss})
                return content_loss

        else:
            if optimizer_idx == 1:
                loss_d = self.disc_step(batch)
                self.log(
                    "Discriminator Loss",
                    loss_d,
                    on_step=True,
                    on_epoch=True,
                    prog_bar=True,
                )
                self.logger.log_metrics({"discriminator_loss": loss_d})
                return loss_d

            if optimizer_idx == 0:
                loss_g = self.gen_step(batch)
                self.log(
                    "Generator Loss", loss_g, on_step=True, on_epoch=True, prog_bar=True
                )
                self.logger.log_metrics({"generator_loss": loss_g})
                return loss_g

    def validation_step(self, batch, batch_idx):
        if self.current_epoch % self.params.save_epoch == 0:
            path = f"./validation_images/Epoch_{self.current_epoch}"
            if not os.path.exists(path):
                os.makedirs(path)
            fake_imgs = []
            img, *_ = batch
            fake_img = self.generator(img)
            fake_img = fake_img.detach().cpu().numpy()
            fake_img = fake_img.transpose(0, 2, 3, 1)
            fake_imgs.append(denormalize_input(fake_img, dtype=np.int16))
            fake_imgs = np.concatenate(fake_imgs, axis=0)

            for i, img in enumerate(fake_imgs):
                save_path = os.path.join(path, f"{i}.jpg")
                cv2.imwrite(save_path, img[..., ::-1])

            save_checkpoint(self.generator, epoch=self.current_epoch, title="generator")
            save_checkpoint(
                self.discriminator, epoch=self.current_epoch, title="discriminator"
            )

    def configure_optimizers(self):
        optimizer_g = optim.Adam(
            self.generator.parameters(), lr=self.params.lr_gen, betas=(0.5, 0.999)
        )
        optimizer_d = optim.Adam(
            self.discriminator.parameters(), lr=self.params.lr_disc, betas=(0.5, 0.999)
        )
        return [optimizer_g, optimizer_d], []

    def train_dataloader(self):
        ds = AnimeDataset(
            data_path=self.params.data_dir, style=self.params.style, isTrain=True
        )
        return torch.utils.data.DataLoader(
            ds,
            batch_size=self.params.batch_size,
            shuffle=True,
            num_workers=6,
        )

    def val_dataloader(self):
        ds = AnimeDataset(
            data_path=self.params.data_dir, style=self.params.style, isTrain=False
        )
        return torch.utils.data.DataLoader(
            ds,
            batch_size=self.params.batch_size,
            num_workers=6,
        )
