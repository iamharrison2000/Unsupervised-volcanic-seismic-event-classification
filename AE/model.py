import torch
from BaseVAE import BaseVAE
from torch import nn
from torch.nn import functional as F
from typing import List, Callable, Union, Any, TypeVar, Tuple
import math

# from torch import tensor as Tensor

Tensor = TypeVar('torch.tensor')

class Encoder_block(nn.Module):

    def __init__(self, in_channel, out_channel, stride=1, Maxpooling=None, **kwargs):
        super(Encoder_block, self).__init__()
        self.Maxpooling = Maxpooling
        
        self.maxpool = nn.MaxPool2d(2,stride=2)
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.Gelu = nn.GELU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel)
        
    def forward(self, x):
        
        if self.Maxpooling:
            x = self.maxpool(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.Gelu(out)
        identity = out
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.Gelu(out)
        
        out = self.conv2(out)
        out = self.bn3(out)
        out += identity
        
        out = self.Gelu(out)
        
        return out
    
    
class Decoder_block(nn.Module):

    def __init__(self, in_channel, out_channel, stride=1, Transpose=None, last_layer=False, **kwargs):
        super(Decoder_block, self).__init__()
        self.Transpose = Transpose
        self.last_layer = last_layer
        self.trans = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=3,
                                        stride = 2, padding=1, output_padding=1)
        self.conv1 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.Gelu = nn.GELU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel)
        
        self.last = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=1, stride=1, padding=0, bias=False)
        
    def forward(self, x):
        # if self.Transpose:
        if self.last_layer:
            out = self.last(x)
            
            return out
        
        if self.Transpose:
            x = self.trans(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.Gelu(out)
        identity = out
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.Gelu(out)
        
        out = self.conv2(out)
        out = self.bn3(out)
        out += identity
        
        out = self.Gelu(out)

        return out
    

    
class BetaVAE(BaseVAE):

    num_iter = 0 # Global static variable to keep track of iterations

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 beta: int = 4,
                 gamma:float = 1000.,
                 max_capacity: int = 25,
                 Capacity_max_iter: int = 1e5,
                 loss_type:str = 'B',
                 Encoder_block = Encoder_block,
                 Decoder_block = Decoder_block, 
                 input_size = (64, 96),
                 **kwargs) -> None:
        super(BetaVAE, self).__init__()
        
        self.row, self.col = input_size
        self.latent_dim = latent_dim
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = Capacity_max_iter
        self.hidden_dims = hidden_dims
        self.in_channels = in_channels
        
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
            
        self.row_min = int(self.row / math.pow(2, len(hidden_dims)-1))
        self.col_min = int(self.col / math.pow(2, len(hidden_dims)-1))
        self.row_col = int(self.row_min * self.col_min)
        
        self.encoder = self._make_encoder(Encoder_block, hidden_dims, latent_dim)
        
        self.fc_mu = nn.Linear(hidden_dims[-1]*self.row_col, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*self.row_col, latent_dim)
        
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * self.row_col)
        
        hidden_dims.reverse()
        self.decoder = self._make_decoder(Decoder_block, hidden_dims, latent_dim)
        
        
    def _make_encoder(self, Encoder_block, hidden_dims, latent_dim):
        Maxpooling = False
        in_channels = self.in_channels
        encoder = []
        for h_dim in hidden_dims:
            encoder.append(
                Encoder_block(in_channels, h_dim, Maxpooling = Maxpooling)
            )
            Maxpooling = True
            in_channels = h_dim

        return nn.Sequential(*encoder)
            
            
    def _make_decoder(self, Decoder_block, hidden_dims, latent_dim):
        Transpose = False
        decoder = []

        in_channels = hidden_dims[0]
        for h_dim in hidden_dims:
            decoder.append(
                Decoder_block(in_channels, h_dim, Transpose = Transpose)
            )
            in_channels = h_dim
            Transpose = True

        #last layer
        decoder.append(
            Decoder_block(in_channels, self.in_channels, Transpose = False, last_layer=True)
        )
        
        return nn.Sequential(*decoder)

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder_input(z)
        # result = result.view(-1, 512, 2, 2)
        result = result.view(-1, max(self.hidden_dims), self.row_min, self.col_min)
        result = self.decoder(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        self.num_iter += 1
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset

        recons_loss =F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        if self.loss_type == 'H': # https://openreview.net/forum?id=Sy2fzU9gl
            loss = recons_loss + self.beta * kld_weight * kld_loss
        elif self.loss_type == 'B': # https://arxiv.org/pdf/1804.03599.pdf
            self.C_max = self.C_max.to(input.device)
            C = torch.clamp(self.C_max/self.C_stop_iter * self.num_iter, 0, self.C_max.data[0])
            loss = recons_loss + self.gamma * kld_weight* (kld_loss - C).abs()
        else:
            raise ValueError('Undefined loss type.')

        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':kld_loss}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]
