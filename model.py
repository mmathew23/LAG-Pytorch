import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, num_filters, activation=nn.ReLU()):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(num_filters, num_filters, 3)
        self.activation = activation
        self.first = nn.Sequential(self.conv1, self.activation)
        self.conv2 = nn.Conv2d(num_filters, num_filters, 3)

    def forward(self, x):
        dx = self.first(x)
        dx = self.conv2(dx)

        return x + dx


class UpsampleBlock(nn.Module):
    def __init__(self, in_filter, out_filter, activation=nn.LeakyReLU(negative_slope=0.2), upsample_module=nn.UpsamplingNearest2d(scale_factor=2)):
        super(UpsampleBlock, self).__init__()

        self.upsample = upsample_module
        self.conv1 = nn.Conv2d(in_filter, out_filter, 3)
        self.conv2 = nn.Conv2d(in_filter, out_filter, 3)
        self.activation = activation
        self.block = nn.Sequential(self.upsample, self.conv1, self.activation, self.conv2, self.activation)

        self.rgb = Conv2d(in_filter, 3, 3)

    def forward(self, x):
        y = self.block(x)

        #we may want to hook into this to grab y for rgb conversion later
        return y, self.rgb(y)

class Generator(nn.Module):
    """
    Generator for LAG
    max_filters: max number of filters in initial layers (should be multiple of 2)
    min_filters: min number of filters towards end layers
    noise_dim: how many noise dimensions to concat to lores input
    blocks: number of residual layers
    """
    def __init__(self, max_filters=256, min_filters=64, upsample_layers=3, noise_dim=64, blocks=8):
        super(Generator, self).__init__()

        # Official LAG tensorflow code uses a custom scaled version of a convolution
        # https://github.com/google-research/lag/blob/e62ef8d32e45dc02315a894e5b223f9427939de0/libml/layers.py#L348
        # For now I use a regular convolution

        # I am also not scaling the output of residuals at the moment, although testing may show it's necessary
        # for convergence

        self.conv1 = nn.Conv2d(3+noise_dim, max_filters, 3)

        #might delete later
        self.relu = nn.ReLU()
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)
        #

        self.residual_blocks = nn.Sequential(*[ResidualBlock(max_filters) for block in range(blocks)])

        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)

        upsample_filters = [max_filters] + [max(min_filters, max_filters >> layer) for layer in range(1, upsample_layers+1)]
        self.upsample_layers = nn.Sequential(*[UpsampleBlock(upsample_filters[layer], upsample_filters[layer+1])])


    def forward(self, x):
        x = self.conv1(x)

        x = self.residual_blocks(x)

        for i, layer in enumerate(self.upsample_layers):
            x, rgb = layer(x)
            if i == 0:
                im = rgb
            else:
                im = self.upsample(im) + rgb

        return im









class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()



