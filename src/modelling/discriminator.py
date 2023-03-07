import torch.nn as nn
from torch.nn.utils import spectral_norm

# from utils import initialize_weights


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


class Discriminator(nn.Module):
    def __init__(self, params):
        super(Discriminator, self).__init__()
        self.bias = False
        channels = 32

        layers = [
            nn.Conv2d(3, channels, kernel_size=3, stride=1, padding=1, bias=self.bias),
            nn.LeakyReLU(0.2, True),
        ]

        for i in range(params.d_layers):
            layers += [
                nn.Conv2d(
                    channels,
                    channels * 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=self.bias,
                ),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(
                    channels * 2,
                    channels * 4,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=self.bias,
                ),
                nn.InstanceNorm2d(channels * 4),
                nn.LeakyReLU(0.2, True),
            ]
            channels *= 4

        layers += [
            nn.Conv2d(
                channels, channels, kernel_size=3, stride=1, padding=1, bias=self.bias
            ),
            nn.InstanceNorm2d(channels),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(channels, 1, kernel_size=3, stride=1, padding=1, bias=self.bias),
        ]

        if params.use_sn:
            for i in range(len(layers)):
                if isinstance(layers[i], nn.Conv2d):
                    layers[i] = spectral_norm(layers[i])

        self.discriminate = nn.Sequential(*layers)

        initialize_weights(self)

    def forward(self, img):
        return self.discriminate(img)
