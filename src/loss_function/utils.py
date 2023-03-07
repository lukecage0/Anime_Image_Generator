import cv2
import numpy as numpy
import torch
import torch.nn as nn

# _rgb_to_yuv_kernel = torch.tensor(
#    [
#        [0.299, -0.14714119, 0.61497538],
#        [0.587, -0.28886916, -0.51496512],
#        [0.114, 0.43601035, -0.10001026],
#    ]
# ).float()

# if torch.cuda.is_available():
#    _rgb_to_yuv_kernel = _rgb_to_yuv_kernel.cuda()

# def save_samples(gen,dataloader):


def denormalize_input(images, dtype=None):
    images = images * 127.5 + 127.5

    if dtype is not None:
        if isinstance(images, torch.Tensor):
            images = images.type(dtype)
        else:
            # numpy.ndarray
            images = images.astype(dtype)

    return images


def initialize_weights(net):
    for m in net.modules():
        try:
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        except Exception as e:
            # print(f'SKip layer {m}, {e}')
            pass


def gram(input):
    """
    Calculate Gram Matrix
    https://pytorch.org/tutorials/advanced/neural_style_tutorial.html#style-loss
    """
    b, c, w, h = input.size()

    x = input.view(b * c, w * h)

    G = torch.mm(x, x.T)

    # normalize by total elements
    return G.div(b * c * w * h)


def rgb_to_yuv(image):
    """
    https://en.wikipedia.org/wiki/YUV
    output: Image of shape (H, W, C) (channel last)
    """
    # -1 1 -> 0 1
    image = (image + 1.0) / 2.0

    yuv_img = torch.tensordot(image, _rgb_to_yuv_kernel, dims=([image.ndim - 3], [0]))

    return yuv_img


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
