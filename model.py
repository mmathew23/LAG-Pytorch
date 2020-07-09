import torch
import torch.nn as nn


def get_upsample_filters(min_filters, max_filters, upsample_layers):
    upsample_filters = [max_filters] + [max(min_filters, max_filters >> layer) for layer in range(1, upsample_layers+1)]
    return upsample_filters


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

        self.rgb = nn.Conv2d(in_filter, 3, 3)

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

        self.noise_dim = noise_dim
        self.conv1 = nn.Conv2d(3+noise_dim, max_filters, 3)

        #might delete later
        self.relu = nn.ReLU()
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)
        #

        self.residual_blocks = nn.Sequential(*[ResidualBlock(max_filters) for block in range(blocks)])

        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)

        #upsample_filters = [max_filters] + [max(min_filters, max_filters >> layer) for layer in range(1, upsample_layers+1)]
        upsample_filters = get_upsample_filters(min_filters, max_filters, upsample_layers)
        self.upsample_layers = nn.Sequential(*[UpsampleBlock(upsample_filters[layer], upsample_filters[layer+1]) for layer, fil in enumerate(upsample_filters[:-1])])


    def forward(self, x, noise_scale=1):
        b, c, h, w = x.shape
        eps = torch.randn(b, self.noise_dim, h, w, device=x.device)
        x = self.conv1(torch.cat([x, eps], dim=1))

        x = self.residual_blocks(x)

        for i, layer in enumerate(self.upsample_layers):
            x, rgb = layer(x)
            if i == 0:
                im = rgb
            else:
                im = self.upsample(im) + rgb

        return im



class SpaceToChannel(nn.Module):
    def __init__(self, n=2):
        super(SpaceToChannel, self).__init__()
        self.n = n

    def forward(self, x):
        n, c, h, w = x.shape
        x = x.reshape(-1, c, h//n, n, w//n, n)
        x = x.permute(0, 1, 3, 5, 2, 4)
        return x.reshape(-1, c*n*n, h//n, w//n)

class DownsampleBlock(nn.Module):
    def __init__(self, stage, filter1, filter2, negative_slope=0.2):
        super(DownsampleBlock, self).__init__()

        self.from_rgb = nn.Conv2d(3, filter1, 3)
        self.leaky = nn.LeakyReLU(negative_slope=negative_slope)

        if stage == 0:
            self.downscale = nn.Identity()
        else:
            self.downscale = nn.AvgPool2d((stage)*2) #we can play with this to see what downscale operator works best

        self.conv1 = nn.Conv2d(filter1, filter1, 3)
        self.space_to_channels = SpaceToChannel()
        self.conv2 = nn.Conv2d(4*filter1, filter2, 3) #4* since we rearrange space onto channels

    def forward(self, x):
        x0, y = x #unpack original image and y

        #initial y in Downsampleblock should be a tensor of 0's
        y += self.from_rgb(self.downscale(x0))
        y = self.leaky(self.conv1(y))
        y = self.space_to_channels(y)
        y = self.leaky(self.conv2(y))
        return x0, y


class ConvWithAct(nn.Module):
    def __init__(self, num_filters, activation, initial=False):
        super(ConvWithAct, self).__init__()

        self.conv1 = nn.Conv2d(num_filters + (3 if initial else 0), num_filters, 3)
        self.activation = activation

        self.mod = nn.Sequential(self.conv1, self.activation)

    def forward(self, x):
        return self.mod(x)


class Discriminator(nn.Module):
    def __init__(self, max_filters=256, min_filters=64, upsample_layers=3, blocks=8):
        super(Discriminator, self).__init__()

        upsample_filters = list(reversed(get_upsample_filters(min_filters, max_filters, upsample_layers)))
        self.downsample_blocks = nn.Sequential(*[DownsampleBlock(stage, upsample_filters[stage], upsample_filters[stage+1]) for stage, fil in enumerate(upsample_filters[:-1])])

        self.blocks = nn.Sequential(*[ConvWithAct(max_filters, nn.LeakyReLU(negative_slope=0.2), initial=True if block == 0 else False) for block in range(blocks)])

        self.center = torch.ones((1, max_filters, 1, 1))
        self.center[:,::2,:,:] = -1
        self.center.requires_grad = False

    def forward(self, x, lowres_x_delta):
        y_0 = torch.zeros(x.shape)

        x0, y = self.downsample_blocks([x, y_0])

        y = torch.cat([y, lowres_x_delta], dim=1)
        y = self.blocks(y)
        return y*self.center




class GAN(object):
    def __init__(self, max_filters=256, min_filters=64, upsample_layers=3, noise_dim=64, blocks=8):
        pass


