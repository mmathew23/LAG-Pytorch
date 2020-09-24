import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torchvision.utils import make_grid
from torch.autograd import Variable
from pathlib import Path
from data import denorm
import os
import re

def get_upsample_filters(min_filters, max_filters, upsample_layers):
    upsample_filters = [max_filters] + [max(min_filters, max_filters >> layer) for layer in range(1, upsample_layers+1)]
    return upsample_filters

def log_loss(writer, n_iter, loss_dreal, loss_dfake, loss_gfake, loss_gmse, loss_gp, loss_disc, loss_gen):
    if writer:
        writer.add_scalar('Loss_G/gfake', loss_gfake, n_iter)
        writer.add_scalar('Loss_G/gmse', loss_gmse, n_iter)
        writer.add_scalar('Loss_G/loss', loss_gen, n_iter)
        writer.add_scalar('Loss_D/dreal', loss_dreal, n_iter)
        writer.add_scalar('Loss_D/dfake', loss_dfake, n_iter)
        writer.add_scalar('Loss_D/gp', loss_gp, n_iter)
        writer.add_scalar('Loss_D/disc', loss_disc, n_iter)

def log_image(writer, n_iter, real, lores, fake, fake_eps):
    if writer:
        writer.add_image('real', denorm(real[0]), n_iter)
        writer.add_image('lores', denorm(lores[0]), n_iter)
        writer.add_image('fake', denorm(fake[0]), n_iter)
        writer.add_image('fake_eps', denorm(fake_eps[0]), n_iter)

def log_samples(writer, n_iter, ims):
    if writer:
        writer.add_image('samples', ims, n_iter)

class ScaledConv2dWithAct(nn.Conv2d):
    """
    simple scaled 2d convolution
    with kaiming style scaling
    and activation
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation='relu'):
        super(ScaledConv2dWithAct, self).__init__(in_channels, out_channels, kernel_size, stride, padding)
        if activation == 'relu':
            self.activation = nn.ReLU()
            slope = 0
        elif activation == 'lrelu':
            slope = 0.2
            self.activation = nn.LeakyReLU(negative_slope=slope)
        else:
            self.activation = nn.Identity()
            slope= 1
        #self.scale =
        self.register_buffer('scale', torch.sqrt(torch.Tensor([2. / ((1+slope*slope)*(kernel_size*kernel_size*in_channels))])))

    def reset_parameters(self):
        init.normal_(self.weight) #normal init with mean 0 and std 1
        init.zeros_(self.bias)

    def forward(self, x):
        #if self.scale.device != self.weight.device:
        #    self.scale = self.scale.to(self.weight.device)
        return self.activation(self.conv2d_forward(x, self.scale*self.weight))

class Conv2dWithAct(nn.Conv2d):
    """
    simple 2d convolution
    with activation
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation='relu'):
        super(Conv2dWithAct, self).__init__(in_channels, out_channels, kernel_size, stride, padding)
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'lrelu':
            slope = 0.2
            self.activation = nn.LeakyReLU(negative_slope=slope)
        else:
            self.activation = nn.Identity()

    def forward(self, x):
        return self.activation(self.conv2d_forward(x, self.weight))

class ResidualBlock(nn.Module):
    def __init__(self, num_filters, activation='relu', conv=ScaledConv2dWithAct, block_scaling=1):
        super(ResidualBlock, self).__init__()

        #self.conv1 = nn.Conv2d(num_filters, num_filters, 3, padding=1)
        self.conv1 = conv(num_filters, num_filters, 3, padding=1, activation=activation)
        #self.activation = activation
        #self.conv2 = nn.Conv2d(num_filters, num_filters, 3, padding=1)
        self.conv2 = conv(num_filters, num_filters, 3, padding=1, activation=None)
        #self.first = nn.Sequential(self.conv1, self.activation, self.conv2)
        self.first = nn.Sequential(self.conv1, self.conv2)

        #block_scaling scales output of the two convolutions
        #I believe the effect is to normalize the output of the conv
        #by the number of residual blocks used to keep training stable
        self.block_scaling = block_scaling

    def forward(self, x):
        dx = self.first(x) / self.block_scaling
        #dx = self.conv2(dx)

        return x + dx


class UpsampleBlock(nn.Module):
    def __init__(self, in_filter, out_filter, activation='lrelu', upsample_module=nn.UpsamplingNearest2d(scale_factor=2), conv=ScaledConv2dWithAct):
        super(UpsampleBlock, self).__init__()

        self.upsample = upsample_module
        #self.conv1 = nn.Conv2d(in_filter, out_filter, 3, padding=1)
        #self.conv2 = nn.Conv2d(out_filter, out_filter, 3, padding=1)
        self.conv1 = conv(in_filter, out_filter, 3, padding=1, activation=activation)
        self.conv2 = conv(out_filter, out_filter, 3, padding=1, activation=activation)
        #self.activation = activation
        #self.block = nn.Sequential(self.upsample, self.conv1, self.activation, self.conv2, self.activation)
        self.block = nn.Sequential(self.upsample, self.conv1, self.conv2)

        self.rgb = conv(out_filter, 3, 3, padding=1, activation=None)

    def forward(self, x):
        #x should be a tuple
        y, rgb_in = x
        y = self.block(y)

        rgb_out = self.rgb(y)

        if rgb_in is None:
            return y, rgb_out
        else:
            up = self.upsample(rgb_in) + rgb_out
            return y, up


class Generator(nn.Module):
    """
    Generator for LAG
    max_filters: max number of filters in initial layers (should be multiple of 2)
    min_filters: min number of filters towards end layers
    noise_dim: how many noise dimensions to concat to lores input
    blocks: number of residual layers
    """
    def __init__(self, max_filters=256, min_filters=64, upsample_layers=3, noise_dim=64,
            blocks=8, conv=ScaledConv2dWithAct, upsample_module=nn.UpsamplingNearest2d(scale_factor=2)):
        super(Generator, self).__init__()

        # Official LAG tensorflow code uses a custom scaled version of a convolution
        # https://github.com/google-research/lag/blob/e62ef8d32e45dc02315a894e5b223f9427939de0/libml/layers.py#L348

        self.noise_dim = noise_dim
        #self.conv1 = nn.Conv2d(3+noise_dim, max_filters, 3, padding=1)
        self.conv1 = conv(3+noise_dim, max_filters, 3, padding=1, activation=None)

        self.residual_blocks = nn.Sequential(*[ResidualBlock(max_filters, conv=conv, block_scaling=blocks) for block in range(blocks)])

        #upsample_filters = [max_filters] + [max(min_filters, max_filters >> layer) for layer in range(1, upsample_layers+1)]
        upsample_filters = get_upsample_filters(min_filters, max_filters, upsample_layers)
        self.upsample_layers = nn.Sequential(*[UpsampleBlock(upsample_filters[layer], upsample_filters[layer+1], conv=conv,
            upsample_module=upsample_module) for layer, fil in enumerate(upsample_filters[:-1])])


    def forward(self, x, noise_scale=1, eps=None):
        b, c, h, w = x.shape
        if eps is None:
            eps = torch.randn(b, self.noise_dim, h, w, device=x.device)
        eps *= noise_scale
        x = self.conv1(torch.cat([x, eps], dim=1))

        x = self.residual_blocks(x)

        y, im = self.upsample_layers((x, None))
        return im



class SpaceToChannel(nn.Module):
    def __init__(self, n=2):
        super(SpaceToChannel, self).__init__()
        self.n = n

    def forward(self, x):
        b, c, h, w = x.shape
        #cant use unfold, seems like 2nd derivative wont work since unfold uses im2col
        #and im2col_backward not implemented
        #return F.unfold(x, self.n, stride=self.n).reshape(b, c*self.n**2, h//self.n, w//self.n)
        x = x.reshape(-1, c, h//self.n, self.n, w//self.n, self.n)
        x = x.permute(0, 1, 3, 5, 2, 4)
        x = x.reshape(-1, c*self.n*self.n, h//self.n, w//self.n)
        return x

class DownsampleBlock(nn.Module):
    def __init__(self, stage, filter1, filter2, negative_slope=0.2, conv=ScaledConv2dWithAct):
        super(DownsampleBlock, self).__init__()

        #self.from_rgb = nn.Conv2d(3, filter1, 3, padding=1)
        self.from_rgb = conv(3, filter1, 3, padding=1, activation='lrelu')
        #self.leaky = nn.LeakyReLU(negative_slope=negative_slope)

        if stage == 0:
            self.downscale = nn.Identity()
        else:
            self.downscale = nn.AvgPool2d(2**stage) #we can play with this to see what downscale operator works best

        #self.conv1 = nn.Conv2d(filter1, filter1, 3, padding=1)
        self.conv1 = conv(filter1, filter1, 3, padding=1, activation='lrelu')
        self.space_to_channels = SpaceToChannel()
        #self.conv2 = nn.Conv2d(4*filter1, filter2, 3, padding=1) #4* since we rearrange space onto channels
        self.conv2 = conv(4*filter1, filter2, 3, padding=1, activation='lrelu') #4* since we rearrange space onto channels
        self.seq1 = nn.Sequential(self.conv1, self.space_to_channels, self.conv2)

    def forward(self, x):
        x0, y = x #unpack original image and y

        #initial y in Downsampleblock should be a tensor of 0's
        y += self.from_rgb(self.downscale(x0))
        y = self.seq1(y)
        return x0, y


#class ConvWithAct(nn.Module):
#    def __init__(self, num_filters, activation, initial=False):
#        super(ConvWithAct, self).__init__()
#
#        self.conv1 = nn.Conv2d(num_filters + (3 if initial else 0), num_filters, 3, padding=1)
#        self.activation = activation
#
#        self.mod = nn.Sequential(self.conv1, self.activation)
#
#    def forward(self, x):
#        return self.mod(x)
#

class Discriminator(nn.Module):
    def __init__(self, max_filters=256, min_filters=64, upsample_layers=3, blocks=8, conv=ScaledConv2dWithAct):
        super(Discriminator, self).__init__()

        self.min_filters = min_filters
        upsample_filters = list(reversed(get_upsample_filters(min_filters, max_filters, upsample_layers)))
        self.filters = upsample_filters
        self.downsample_blocks = nn.Sequential(*[DownsampleBlock(stage, upsample_filters[stage], upsample_filters[stage+1], conv=conv) for stage, fil in enumerate(upsample_filters[:-1])])

        #self.blocks = nn.Sequential(*[ConvWithAct(max_filters, nn.LeakyReLU(negative_slope=0.2), initial=True if block == 0 else False) for block in range(blocks)])
        self.blocks = nn.Sequential(*[conv(max_filters + (3 if block==0 else 0), max_filters, 3, padding=1, activation='lrelu') for block in range(blocks)])

        center = torch.ones((1, max_filters, 1, 1))
        center[:,::2,:,:] = -1
        #self.center.requires_grad = False #I don't think this needs to be here
        self.register_buffer('center', center)

    def forward(self, x, lowres_x_delta):
        b, c, h, w = x.shape
        y_0 = torch.zeros((b, self.filters[0], h, w), device=x.device)

        x0, y = self.downsample_blocks([x, y_0])

        y = torch.cat([y, lowres_x_delta], dim=1)
        y = self.blocks(y)
        return y*self.center




class GAN(object):
    def __init__(self, max_filters=256, min_filters=64, upsample_layers=3, noise_dim=64, blocks=8, device_ids=0,
            models_dir='./models/', results_dir='./results/', log_writer=None, model_name='test',
            save_every=10000, print_every=500, log_every=500, conv_type='regular', grad_mean=False,
            upsample_type='nearest', straight_through_round=True, samples=None):
        self.device_ids = device_ids
        self.device = f'cuda:{device_ids[0]}'
        self.noise_dim = noise_dim
        self.upsample_layers = upsample_layers
        self.models_dir = Path(models_dir)
        self.results_dir = Path(results_dir)
        self.log_writer=log_writer
        self.model_name = model_name
        self.save_every = save_every
        self.print_every = print_every
        self.log_every = log_every
        self.grad_mean = grad_mean
        self.resume_number = 0
        self.straight_through = straight_through_round
        self.samples = samples
        if conv_type == 'regular':
            self.conv= Conv2dWithAct
        elif conv_type == 'scaled':
            self.conv= ScaledConv2dWithAct
        else:
            raise TypeError("conv_type must be one of 'regular' or 'scaled'")

        #upsample type must be one of pytorch modes
        if upsample_type[-6:]=='linear' or upsample_type[-5:]=='cubic':
            upsample = nn.Upsample(scale_factor=2, mode=upsample_type, align_corners=True)
        else:
            upsample = nn.Upsample(scale_factor=2, mode=upsample_type)


        self.G = Generator( max_filters=max_filters, min_filters=min_filters,
                upsample_layers=upsample_layers, noise_dim=noise_dim, blocks=blocks,
                conv=self.conv, upsample_module=upsample)
        self.D = Discriminator(max_filters=max_filters, min_filters=min_filters,
                upsample_layers=upsample_layers, blocks=blocks, conv=self.conv)

        if len(self.device_ids) > 1:
            self.G = nn.DataParallel(self.G, device_ids=self.device_ids)
            self.D = nn.DataParallel(self.D, device_ids=self.device_ids)

        self.G.to(self.device)
        self.D.to(self.device)

        self.downscale = nn.AvgPool2d(2**self.upsample_layers)
        #currently no ema

    def straight_through_round(self, x, r=127.5/4):
        if not self.straight_through:
            return x
        xr = torch.round(x*r)/r
        dxr = xr - x
        dxr = dxr.detach()
        return dxr + x

    def save(self, num):
        torch.save(self.G.state_dict(), self.models_dir / self.model_name / f'Gmodel_{num}.pt')
        torch.save(self.D.state_dict(), self.models_dir / self.model_name / f'Dmodel_{num}.pt')
        if not (self.samples is None):
            torch.save(self.samples, self.models_dir / self.model_name / 'samples.pt')

    def load(self, num):
        if num == -1:
            files = os.listdir(self.models_dir / self.model_name)
            nums_match = [re.search("[0-9]+", f) for f in files]
            nums = [int(match[0]) for match in nums_match if match is not None]
            if len(nums):
                num = max(nums)
                self.resume_number = int(num)
            else:
                #we haven't saved yet so don't attempt to load
                return

        G_name = self.models_dir / self.model_name / f'Gmodel_{num}.pt'
        D_name = self.models_dir / self.model_name / f'Dmodel_{num}.pt'
        self.G.load_state_dict(torch.load(G_name))
        self.D.load_state_dict(torch.load(D_name))
        if os.path.exists(self.models_dir / self.model_name / 'samples.pt'):
            print('loading samples')
            self.samples = torch.load(self.models_dir / self.model_name / 'samples.pt')

    def gen_samples(self):
        with torch.no_grad():
            self.G.eval()
            lores, hires = self.samples
            lores = lores.cuda(self.device)
            b, c, h, w = lores.shape
            noise = torch.zeros(b, self.noise_dim, h, w, device=self.device)
            fake = self.G(lores, noise)
            self.G.train()
            lores = F.interpolate(lores, scale_factor=2**self.upsample_layers, mode='nearest')
            lores = lores.to('cpu')
            fake = fake.to('cpu')
            ims = torch.cat((lores, fake, hires), dim=-1)
            ims = make_grid(ims, nrow=2, normalize=True, range=(-1,1))
        return ims

    def train(self, dataloader, epochs, lr, wass_target, mse_weight, ttur, save_every=1):
        #torch.backends.cudnn.benchmark = True
        #torch.autograd.set_detect_anomaly(True)
        self.G.train()
        self.D.train()
        #epsilon value is the tensorflow default, could probably take this out
        #G_opt = torch.optim.Adam(self.G.parameters(), lr=lr, betas=(0.9, 0.99), eps=1e-5)
        #D_opt = torch.optim.Adam(self.D.parameters(), lr=lr, betas=(0.9, 0.99), eps=1e-5)
        #LAG tf implementation uses 0 and .99
        G_opt = torch.optim.Adam(self.G.parameters(), lr=lr, betas=(0, 0.99), eps=1e-8)
        D_opt = torch.optim.Adam(self.D.parameters(), lr=lr, betas=(0, 0.99), eps=1e-8)
        #iterate through loader
        n_iter = 0
        for epoch in range(epochs):
            for i, batch in enumerate(dataloader):
                lores, real = batch
                lores = lores.cuda(self.device)
                real = real.cuda(self.device)

                #gen noise
                b, c, h, w = lores.shape
                eps = torch.randn(b, self.noise_dim, h, w, device=self.device)

                #train G
                G_opt.zero_grad()
                #gen fake no noise
                fake = self.G(lores, eps=torch.zeros_like(eps))
                #gen fake plus noise
                fake_eps = self.G(lores, eps=eps)

                #downscale fake no noise and fake with noise
                lores_fake = self.downscale(fake)
                lores_fake_eps = self.downscale(fake_eps)

                latent_real = self.D(real, self.straight_through_round(torch.abs(lores-lores)))
                latent_fake = self.D(fake, self.straight_through_round(torch.abs(lores - lores_fake)))
                latent_fake_eps = self.D(fake_eps, self.straight_through_round(torch.abs(lores - lores_fake_eps)))
                #loss for G
                loss_gfake = -torch.mean(latent_fake_eps)
                #detach latent_real?
                loss_gmse = F.mse_loss(latent_fake, latent_real.detach())
                loss_gen = loss_gfake + mse_weight * loss_gmse



                #G backward and opt step
                loss_gen.backward()
                G_opt.step()
                loss_gen = float(loss_gen)

                #Train disc
                D_opt.zero_grad()


                #xs = Variable(mixed, requires_grad=True)
                if self.grad_mean:
                    #disc mixed for gradient penalty
                    uniform_mix = torch.rand(b, 1, 1, 1, device=torch.device(self.device)) #.cuda(self.device)
                    mixed = real + uniform_mix * (fake_eps.detach() - real) #detach fake_eps, is needed?
                    xs = Variable(mixed, requires_grad=True)
                    mixed_round = self.straight_through_round(torch.abs(lores - self.downscale(xs)))
                    ys = torch.mean(self.D(xs, mixed_round))
                    grad = torch.autograd.grad(ys, xs, create_graph=True, retain_graph=True)[0]
                    grad= torch.sqrt(torch.sum(grad*grad, dim=[1,2,3])+1e-8)
                    loss_gp = 10*torch.mean((grad- wass_target)**2) * wass_target ** -2
                else:
                    #disc mixed for gradient penalty
                    uniform_mix = torch.rand(b, 1, 1, 1, device=torch.device(self.device)) #.cuda(self.device)
                    mixed = real + uniform_mix * (fake_eps.detach() - real) #detach fake_eps, is needed?
                    xs = Variable(mixed, requires_grad=True)
                    mixed_round = self.straight_through_round(torch.abs(lores - self.downscale(xs)))
                    ys = torch.sum(torch.mean(self.D(xs, mixed_round), dim=1))
                    grad = torch.autograd.grad(ys, xs, create_graph=True, retain_graph=True)[0]
                    grad= torch.sqrt(torch.mean(grad*grad, dim=[1,2,3])+1e-8)
                    loss_gp = 10*torch.mean((grad- wass_target)*(grad-wass_target)) * wass_target ** -2

                latent_real = self.D(real, self.straight_through_round(torch.abs(lores-lores)))
                latent_fake_eps = self.D(fake_eps.detach(), self.straight_through_round(torch.abs(lores - lores_fake_eps.detach())))

                #calc losses for D
                loss_dreal = -torch.mean(latent_real)
                loss_dfake = torch.mean(latent_fake_eps)
                loss_disc = ttur * (loss_dreal + loss_dfake + loss_gp)


                #D opt step
                loss_disc.backward()
                loss_disc = float(loss_disc)
                D_opt.step()


                n_iter = (epoch)*len(dataloader)+i+self.resume_number
                if (i-1)%self.print_every == 0:

                    print(f'Epoch: {epoch}. Loss: GLoss: {loss_gen} DLoss: {loss_disc} gfake: {loss_gfake} gmse: {loss_gmse} dreal: {loss_dreal} dfake: {loss_dfake} gp: {loss_gp}')
                if (i-1)%self.log_every == 0:
                    log_loss(self.log_writer, n_iter, loss_dreal, loss_dfake, loss_gfake, loss_gmse, loss_gp, loss_disc, loss_gen)
                    log_image(self.log_writer, n_iter, real, lores, fake, fake_eps)
                    log_samples(self.log_writer, n_iter, self.gen_samples())

                if n_iter%self.save_every == 0 and (i!=0):
                    self.save(n_iter)
        return n_iter

