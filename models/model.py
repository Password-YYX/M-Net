r""" The proposed M-Net
"""

import torch
import torch.nn as nn
from collections import OrderedDict

from utils import logger

__all__ = ["model_mnet"]


class MlpBlock(nn.Module):
    def __init__(self, hidden_dim, mlp_dim, dropout=0.):
        super(MlpBlock, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.mlp(x)


class MixerBlock(nn.Module):
    def __init__(self, num_tokens, hidden_dim, tokens_mlp_dim, channels_mlp_dim, dropout=0.):
        super(MixerBlock, self).__init__()
        self.ln_token = nn.LayerNorm(hidden_dim)
        self.token_mix = MlpBlock(num_tokens, tokens_mlp_dim, dropout)
        self.ln_channel = nn.LayerNorm(hidden_dim)
        self.channel_mix = MlpBlock(hidden_dim, channels_mlp_dim, dropout)

    def forward(self, x):
        out = self.ln_token(x).transpose(1, 2)
        x = x + self.token_mix(out).transpose(1, 2)
        out = self.ln_channel(x)
        x = x + self.channel_mix(out)
        return x


class MlpMixer_Encoder(nn.Module):
    def __init__(self, num_blocks, hidden_dim, tokens_mlp_dim, channels_mlp_dim, patch_size=None, image_size=None,
                 dropout=0.):
        super(MlpMixer_Encoder, self).__init__()
        if patch_size is None:
            patch_size = [32, 1]
        if image_size is None:
            image_size = [32, 32]
        assert image_size[0] % patch_size[0] == 0, 'Image dimensions must be divisible by the patch size.'
        assert image_size[1] % patch_size[1] == 0, 'Image dimensions must be divisible by the patch size.'
        num_tokens = (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1])

        self.patch_emb = nn.Conv2d(2, hidden_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.mlp = nn.Sequential(
            *[MixerBlock(num_tokens, hidden_dim, tokens_mlp_dim, channels_mlp_dim, dropout) for _ in range(num_blocks)])
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = self.patch_emb(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.mlp(x)
        x = self.ln(x)
        return x


class MlpMixer_Decoder(nn.Module):
    def __init__(self, num_blocks1, num_blocks2, num_blocks3, hidden_dim, tokens_mlp_dim, channels_mlp_dim,
                 patch_size=None, image_size=None, dropout=0.):
        super(MlpMixer_Decoder, self).__init__()
        if patch_size is None:
            patch_size = [32, 1]
        if image_size is None:
            image_size = [32, 32]
        assert image_size[0] % patch_size[0] == 0, 'Image dimensions must be divisible by the patch size.'
        assert image_size[1] % patch_size[1] == 0, 'Image dimensions must be divisible by the patch size.'
        num_tokens = (image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1])

        self.patch_emb = nn.Conv2d(2, hidden_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.mlp1 = nn.Sequential(
            *[MixerBlock(num_tokens, hidden_dim, tokens_mlp_dim, channels_mlp_dim, dropout) for _ in range(num_blocks1)])
        self.mlp2 = nn.Sequential(
            *[MixerBlock(num_tokens, hidden_dim, tokens_mlp_dim, channels_mlp_dim, dropout) for _ in range(num_blocks2)])
        self.mlp3 = nn.Sequential(
            *[MixerBlock(num_tokens, hidden_dim, tokens_mlp_dim, channels_mlp_dim, dropout) for _ in range(num_blocks3)])
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = self.patch_emb(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.mlp3(x)
        x = self.ln(x)
        return x


class MNet(nn.Module):
    def __init__(self, reduction, expansion):
        super(MNet, self).__init__()
        total_size, in_channel, w, h = 2048, 2, 32, 32
        logger.info(f'reduction={reduction}')
        self.encoder = MlpMixer_Encoder(1, 64, 32, 64)
        self.encoder_fc = nn.Linear(total_size, total_size // reduction)

        self.decoder_fc = nn.Linear(total_size // reduction, total_size)
        self.decoder = MlpMixer_Decoder(1*expansion, 1*expansion, 1*expansion, 64, 32, 64)
        self.sig = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        n, c, h, w = x.detach().size()
        out = self.encoder(x)
        out = self.encoder_fc(out.contiguous().view(n, -1))

        out = self.decoder_fc(out).view(n, c, h, w)
        out= self.decoder(out)
        out = out.view(n, c, h, w)
        out = self.sig(out)

        return out


def model_mnet(reduction, expansion):
    r""" Create a proposed M-Net.

    :param reduction: the reciprocal of compression ratio
    :return: an instance of M-Net
    """

    model = MNet(reduction=reduction, expansion=expansion)
    return model
