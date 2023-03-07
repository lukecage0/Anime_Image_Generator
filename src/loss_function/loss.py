import torch
import torch.nn as nn

from loss_function.utils import gram, rgb_to_yuv


class ColorLoss(nn.Module):
    def __init__(self):
        super(ColorLoss, self).__init__()
        self.l1 = nn.L1Loss()
        self.huber = nn.SmoothL1Loss()

    def forward(self, image, image_g):
        image = rgb_to_yuv(image)
        image_g = rgb_to_yuv(image_g)

        # After convert to yuv, both images have channel last

        return (
            self.l1(image[:, :, :, 0], image_g[:, :, :, 0])
            + self.huber(image[:, :, :, 1], image_g[:, :, :, 1])
            + self.huber(image[:, :, :, 2], image_g[:, :, :, 2])
        )


class AnimeGANLoss:
    def __init__(self, params):
        # self.vgg = vgg
        self.params = params
        self.l1 = nn.L1Loss()
        self.huber = nn.SmoothL1Loss()

    def vgg_content_loss(self, vgg, img, reconstruction):
        feat = vgg(img)
        re_feat = vgg(reconstruction)
        return self.l1(feat, re_feat)

    def color_loss(self, image, image_g, rgb_to_yuv):
        image = rgb_to_yuv(image)
        image_g = rgb_to_yuv(image_g)

        # After convert to yuv, both images have channel last

        return (
            self.l1(image[:, :, :, 0], image_g[:, :, :, 0])
            + self.huber(image[:, :, :, 1], image_g[:, :, :, 1])
            + self.huber(image[:, :, :, 2], image_g[:, :, :, 2])
        )

    def compute_generator_loss(
        self,
        vgg,
        rgb_to_yuv,
        fake_img,
        img,
        fake_logits,
        anime_gray,
    ):
        fake_feat = vgg(fake_img)
        anime_feat = vgg(anime_gray)
        img_feat = vgg(img)

        # To-Do Implement
        return [
            self.params.wadv_g * self.adv_loss_generator(fake_logits),
            self.params.wcontent * self.l1(img_feat, fake_feat),
            self.params.wgram * self.l1(gram(anime_feat), gram(fake_feat)),
            self.params.wcolor * self.color_loss(img, fake_img),
        ]

    def compute_discriminator_loss(
        self, fake_img_d, real_anime_d, real_anime_gray_d, real_anime_smooth_gray_d
    ):
        return self.params.wadv_d * (
            self.adv_loss_d_real(real_anime_d)
            + self.adv_loss_d_fake(fake_img_d)
            + self.adv_loss_d_fake(real_anime_gray_d)
            + 0.2 * self.adv_loss_d_fake(real_anime_smooth_gray_d)
        )

    @staticmethod
    def adv_loss_generator(pred):
        return torch.mean(torch.square(pred - 1))

    @staticmethod
    def adv_loss_d_fake(pred):
        return torch.mean(torch.square(pred))

    @staticmethod
    def adv_loss_d_real(pred):
        return torch.mean(torch.square(pred - 1.0))
