import torch.nn as nn


class ConvNormLReLU(nn.Sequential):
    def __init__(
        self,
        in_ch,
        out_ch,
        kernel_size=3,
        stride=1,
        padding=1,
        pad_mode="reflect",
        groups=1,
        bias=False,
    ):

        pad_layer = {
            "zero": nn.ZeroPad2d,
            "same": nn.ReplicationPad2d,
            "reflect": nn.ReflectionPad2d,
        }
        if pad_mode not in pad_layer:
            raise NotImplementedError

        super(ConvNormLReLU, self).__init__(
            pad_layer[pad_mode](padding),
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=0,
                groups=groups,
                bias=bias,
            ),
            nn.GroupNorm(num_groups=1, num_channels=out_ch, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )


class InvertedResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, expansion_ratio=2):
        super(InvertedResBlock, self).__init__()

        self.use_res_connect = in_ch == out_ch
        bottleneck = int(round(in_ch * expansion_ratio))
        layers = []
        if expansion_ratio != 1:
            layers.append(ConvNormLReLU(in_ch, bottleneck, kernel_size=1, padding=0))

        # dw
        layers.append(
            ConvNormLReLU(bottleneck, bottleneck, groups=bottleneck, bias=True)
        )
        # pw
        layers.append(
            nn.Conv2d(bottleneck, out_ch, kernel_size=1, padding=0, bias=False)
        )
        layers.append(nn.GroupNorm(num_groups=1, num_channels=out_ch, affine=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        out = self.layers(input)
        if self.use_res_connect:
            out = input + out
        return out
