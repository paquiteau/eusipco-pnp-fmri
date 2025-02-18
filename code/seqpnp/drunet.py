from __future__ import annotations
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np

from deepinv.models.utils import get_weights_url, test_onesplit, test_pad

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def load_drunet_mri(
    filename: Path | str, device="cpu", dim=3, norm_factor=1.0
) -> ComplexDenoiser:
    drunet = DRUNet(in_channels=2, out_channels=2, dim=dim, pretrained=None)
    state_dict = torch.load(filename, weights_only=True)
    drunet.load_state_dict(state_dict)
    drunet = drunet.to(device)
    return ComplexDenoiser(drunet, norm_factor=norm_factor).to(device)


class DRUNet(nn.Module):
    r"""
    DRUNet denoiser network.

    The network architecture is based on the paper
    `Learning deep CNN denoiser prior for image restoration <https://arxiv.org/abs/1704.03264>`_,
    and has a U-Net like structure, with convolutional blocks in the encoder and decoder parts.

    The network takes into account the noise level of the input image, which is encoded as an additional input channel.

    A pretrained network for (in_channels=out_channels=1 or in_channels=out_channels=3)
    can be downloaded via setting ``pretrained='download'``.

    :param int in_channels: number of channels of the input.
    :param int out_channels: number of channels of the output.
    :param list nc: number of convolutional layers.
    :param int nb: number of convolutional blocks per layer.
    :param int nf: number of channels per convolutional layer.
    :param str act_mode: activation mode, "R" for ReLU, "L" for LeakyReLU "E" for ELU and "S" for Softplus.
    :param str downsample_mode: Downsampling mode, "avgpool" for average pooling, "maxpool" for max pooling, and
        "strideconv" for convolution with stride 2.
    :param str upsample_mode: Upsampling mode, "convtranspose" for convolution transpose, "pixelsuffle" for pixel
        shuffling, and "upconv" for nearest neighbour upsampling with additional convolution.
    :param str, None pretrained: use a pretrained network. If ``pretrained=None``, the weights will be initialized at random
        using Pytorch's default initialization. If ``pretrained='download'``, the weights will be downloaded from an
        online repository (only available for the default architecture with 3 or 1 input/output channels).
        Finally, ``pretrained`` can also be set as a path to the user's own pretrained weights.
        See :ref:`pretrained-weights <pretrained-weights>` for more details.
    :param bool train: training or testing mode.
    :param str device: gpu or cpu.

    """

    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        nc=[64, 128, 256, 512],
        nb=4,
        act_mode="R",
        downsample_mode="strideconv",
        upsample_mode="convtranspose",
        pretrained="download",
        train=False,
        device=None,
        blind=True,
        dim=2,
    ):
        super(DRUNet, self).__init__()

        self.blind = blind
        in_channels = in_channels + 1  # accounts for the input noise channel
        self.dim = dim

        self.m_head = conv(in_channels, nc[0], bias=False, mode="C", dim=dim)

        # downsample
        if downsample_mode == "avgpool":
            downsample_block = downsample_avgpool
        elif downsample_mode == "maxpool":
            downsample_block = downsample_maxpool
        elif downsample_mode == "strideconv":
            downsample_block = downsample_strideconv
        else:
            raise NotImplementedError(
                "downsample mode [{:s}] is not found".format(downsample_mode)
            )

        self.m_down1 = sequential(
            *[
                ResBlock(nc[0], nc[0], bias=False, mode="C" + act_mode + "C", dim=dim)
                for _ in range(nb)
            ],
            downsample_block(nc[0], nc[1], bias=False, mode="2", dim=dim),
        )
        self.m_down2 = sequential(
            *[
                ResBlock(nc[1], nc[1], bias=False, mode="C" + act_mode + "C", dim=dim)
                for _ in range(nb)
            ],
            downsample_block(nc[1], nc[2], bias=False, mode="2", dim=dim),
        )
        self.m_down3 = sequential(
            *[
                ResBlock(nc[2], nc[2], bias=False, mode="C" + act_mode + "C", dim=dim)
                for _ in range(nb)
            ],
            downsample_block(nc[2], nc[3], bias=False, mode="2", dim=dim),
        )

        self.m_body = sequential(
            *[
                ResBlock(nc[3], nc[3], bias=False, mode="C" + act_mode + "C", dim=dim)
                for _ in range(nb)
            ]
        )

        # upsample
        if upsample_mode == "upconv":
            upsample_block = upsample_upconv
        elif upsample_mode == "pixelshuffle":
            upsample_block = upsample_pixelshuffle
        elif upsample_mode == "convtranspose":
            upsample_block = upsample_convtranspose
        else:
            raise NotImplementedError(
                "upsample mode [{:s}] is not found".format(upsample_mode)
            )

        self.m_up3 = sequential(
            upsample_block(nc[3], nc[2], bias=False, mode="2", dim=dim),
            *[
                ResBlock(nc[2], nc[2], bias=False, mode="C" + act_mode + "C", dim=dim)
                for _ in range(nb)
            ],
        )
        self.m_up2 = sequential(
            upsample_block(nc[2], nc[1], bias=False, mode="2", dim=dim),
            *[
                ResBlock(nc[1], nc[1], bias=False, mode="C" + act_mode + "C", dim=dim)
                for _ in range(nb)
            ],
        )
        self.m_up1 = sequential(
            upsample_block(nc[1], nc[0], bias=False, mode="2", dim=dim),
            *[
                ResBlock(nc[0], nc[0], bias=False, mode="C" + act_mode + "C", dim=dim)
                for _ in range(nb)
            ],
        )

        self.m_tail = conv(nc[0], out_channels, bias=False, mode="C", dim=dim)
        if pretrained is not None:
            if pretrained == "download":
                if in_channels == 4:
                    name = "drunet_deepinv_color_finetune_22k.pth"
                elif in_channels == 2:
                    name = "drunet_deepinv_gray_finetune_26k.pth"
                url = get_weights_url(model_name="drunet", file_name=name)
                ckpt_drunet = torch.hub.load_state_dict_from_url(
                    url, map_location=lambda storage, loc: storage, file_name=name
                )
            else:
                ckpt_drunet = torch.load(
                    pretrained, map_location=lambda storage, loc: storage
                )

                ckpt_new = update_keyvals(ckpt_drunet, self.state_dict())

            self.load_state_dict(ckpt_new, strict=True)
        else:
            self.apply(weights_init_drunet)

        if not train:
            self.eval()
            for _, v in self.named_parameters():
                v.requires_grad = False

        if device is not None:
            self.to(device)

    def forward_unet(self, x0):
        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x + x4)
        x = self.m_up2(x + x3)
        x = self.m_up1(x + x2)
        x = self.m_tail(x + x1)
        return x

    def forward(self, x, sigma):
        r"""
        Run the denoiser on image with noise level :math:`\sigma`.

        :param torch.Tensor x: noisy image
        :param float, torch.Tensor sigma: noise level. If ``sigma`` is a float, it is used for all images in the batch.
            If ``sigma`` is a tensor, it must be of shape ``(batch_size,)``.
        """
        if isinstance(sigma, torch.Tensor):
            if self.dim == 2:
                if sigma.ndim > 0:
                    noise_level_map = sigma.view(x.size(0), 1, 1, 1)
                    noise_level_map = noise_level_map.expand(
                        -1, 1, x.size(2), x.size(3)
                    )
                else:
                    noise_level_map = torch.ones(
                        (x.size(0), 1, x.size(2), x.size(3)), device=x.device
                    ) * sigma[None, None, None, None].to(x.device)
            elif self.dim == 3:
                if sigma.ndim > 0:
                    noise_level_map = sigma.view(x.size(0), 1, 1, 1, 1)
                    noise_level_map = noise_level_map.expand(
                        -1, 1, x.size(2), x.size(3), x.size(4)
                    )
                else:
                    noise_level_map = torch.ones(
                        (x.size(0), 1, x.size(2), x.size(3), x.size(4)), device=x.device
                    ) * sigma[None, None, None, None, None].to(x.device)
        else:
            # TODO: clean
            if self.dim == 2:
                noise_level_map = (
                    torch.ones((x.size(0), 1, x.size(2), x.size(3)), device=x.device)
                    * sigma
                )
            elif self.dim == 3:
                noise_level_map = (
                    torch.ones(
                        (x.size(0), 1, x.size(2), x.size(3), x.size(4)), device=x.device
                    )
                    * sigma
                )
        x = torch.cat((x, noise_level_map), 1)
        if self.training or (
            x.size(2) % 8 == 0
            and x.size(3) % 8 == 0
            and x.size(2) > 31
            and x.size(3) > 31
        ):
            x = self.forward_unet(x)
        elif x.size(2) < 32 or x.size(3) < 32:
            x = test_pad(self.forward_unet, x, modulo=16)
        else:
            x = test_onesplit(self.forward_unet, x, refield=64)
        return x


"""
Functional blocks below
"""
from collections import OrderedDict
import torch
import torch.nn as nn


"""
# --------------------------------------------
# Advanced nn.Sequential
# https://github.com/xinntao/BasicSR
# --------------------------------------------
"""


def sequential(*args):
    """Advanced nn.Sequential.
    Args:
        nn.Sequential, nn.Module
    Returns:
        nn.Sequential
    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError("sequential does not support OrderedDict input.")
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)


"""
# --------------------------------------------
# Useful blocks
# https://github.com/xinntao/BasicSR
# --------------------------------
# conv + normaliation + relu (conv)
# (PixelUnShuffle)
# (ConditionalBatchNorm2d)
# concat (ConcatBlock)
# sum (ShortcutBlock)
# resblock (ResBlock)
# Channel Attention (CA) Layer (CALayer)
# Residual Channel Attention Block (RCABlock)
# Residual Channel Attention Group (RCAGroup)
# Residual Dense Block (ResidualDenseBlock_5C)
# Residual in Residual Dense Block (RRDB)
# --------------------------------------------
"""


# --------------------------------------------
# return nn.Sequantial of (Conv + BN + ReLU)
# --------------------------------------------
def conv(
    in_channels=64,
    out_channels=64,
    kernel_size=3,
    stride=1,
    padding=1,
    bias=True,
    mode="CBR",
    negative_slope=0.2,
    dim=2,
):
    L = []
    for t in mode:
        if t == "C":
            if dim == 2:
                conv = nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                )
            elif dim == 3:
                conv = nn.Conv3d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                )
            L.append(conv)
        elif t == "T":
            if dim == 2:
                conv = nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                )
            elif dim == 3:
                conv = nn.ConvTranspose3d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias,
                )
            L.append(conv)
        elif t == "B":
            L.append(nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-04, affine=True))
        elif t == "I":
            L.append(nn.InstanceNorm2d(out_channels, affine=True))
        elif t == "R":
            L.append(nn.ReLU(inplace=True))
        elif t == "r":
            L.append(nn.ReLU(inplace=False))
        elif t == "L":
            L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=True))
        elif t == "l":
            L.append(nn.LeakyReLU(negative_slope=negative_slope, inplace=False))
        elif t == "E":
            L.append(nn.ELU(inplace=False))
        elif t == "s":
            L.append(nn.Softplus())
        elif t == "2":
            L.append(nn.PixelShuffle(upscale_factor=2))
        elif t == "3":
            L.append(nn.PixelShuffle(upscale_factor=3))
        elif t == "4":
            L.append(nn.PixelShuffle(upscale_factor=4))
        elif t == "U":
            L.append(nn.Upsample(scale_factor=2, mode="nearest"))
        elif t == "u":
            L.append(nn.Upsample(scale_factor=3, mode="nearest"))
        elif t == "v":
            L.append(nn.Upsample(scale_factor=4, mode="nearest"))
        elif t == "M":
            L.append(nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=0))
        elif t == "A":
            L.append(nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=0))
        else:
            raise NotImplementedError("Undefined type: ")
    return sequential(*L)


# --------------------------------------------
# Res Block: x + conv(relu(conv(x)))
# --------------------------------------------
class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels=64,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=True,
        mode="CRC",
        negative_slope=0.2,
        dim=2,
    ):
        super(ResBlock, self).__init__()

        assert in_channels == out_channels, "Only support in_channels==out_channels."
        if mode[0] in ["R", "L"]:
            mode = mode[0].lower() + mode[1:]

        self.res = conv(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias,
            mode,
            negative_slope,
            dim=dim,
        )

    def forward(self, x):
        res = self.res(x)  # TODO not gona work
        return x + res


"""
# --------------------------------------------
# Upsampler
# Kai Zhang, https://github.com/cszn/KAIR
# --------------------------------------------
# upsample_pixelshuffle
# upsample_upconv
# upsample_convtranspose
# --------------------------------------------
"""


# --------------------------------------------
# conv + subp (+ relu)
# --------------------------------------------
def upsample_pixelshuffle(
    in_channels=64,
    out_channels=3,
    kernel_size=3,
    stride=1,
    padding=1,
    bias=True,
    mode="2R",
    negative_slope=0.2,
):
    assert len(mode) < 4 and mode[0] in [
        "2",
        "3",
        "4",
    ], "mode examples: 2, 2R, 2BR, 3, ..., 4BR."
    up1 = conv(
        in_channels,
        out_channels * (int(mode[0]) ** 2),
        kernel_size,
        stride,
        padding,
        bias,
        mode="C" + mode,
        negative_slope=negative_slope,
    )
    return up1


# --------------------------------------------
# nearest_upsample + conv (+ R)
# --------------------------------------------
def upsample_upconv(
    in_channels=64,
    out_channels=3,
    kernel_size=3,
    stride=1,
    padding=1,
    bias=True,
    mode="2R",
    negative_slope=0.2,
    dim=2,
):
    assert len(mode) < 4 and mode[0] in [
        "2",
        "3",
        "4",
    ], "mode examples: 2, 2R, 2BR, 3, ..., 4BR"
    if mode[0] == "2":
        uc = "UC"
    elif mode[0] == "3":
        uc = "uC"
    elif mode[0] == "4":
        uc = "vC"
    mode = mode.replace(mode[0], uc)
    up1 = conv(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        bias,
        mode=mode,
        negative_slope=negative_slope,
        dim=dim,
    )
    return up1


# --------------------------------------------
# convTranspose (+ relu)
# --------------------------------------------
def upsample_convtranspose(
    in_channels=64,
    out_channels=3,
    kernel_size=2,
    stride=2,
    padding=0,
    bias=True,
    mode="2R",
    negative_slope=0.2,
    dim=2,
):
    assert len(mode) < 4 and mode[0] in [
        "2",
        "3",
        "4",
    ], "mode examples: 2, 2R, 2BR, 3, ..., 4BR."
    kernel_size = int(mode[0])
    stride = int(mode[0])
    mode = mode.replace(mode[0], "T")
    up1 = conv(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        bias,
        mode,
        negative_slope,
        dim=dim,
    )
    return up1


"""
# --------------------------------------------
# Downsampler
# Kai Zhang, https://github.com/cszn/KAIR
# --------------------------------------------
# downsample_strideconv
# downsample_maxpool
# downsample_avgpool
# --------------------------------------------
"""


# --------------------------------------------
# strideconv (+ relu)
# --------------------------------------------
def downsample_strideconv(
    in_channels=64,
    out_channels=64,
    kernel_size=2,
    stride=2,
    padding=0,
    bias=True,
    mode="2R",
    negative_slope=0.2,
    dim=2,
):
    assert len(mode) < 4 and mode[0] in [
        "2",
        "3",
        "4",
    ], "mode examples: 2, 2R, 2BR, 3, ..., 4BR."
    kernel_size = int(mode[0])
    stride = int(mode[0])
    mode = mode.replace(mode[0], "C")
    down1 = conv(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        bias,
        mode,
        negative_slope,
        dim=dim,
    )
    return down1


# --------------------------------------------
# maxpooling + conv (+ relu)
# --------------------------------------------
def downsample_maxpool(
    in_channels=64,
    out_channels=64,
    kernel_size=3,
    stride=1,
    padding=0,
    bias=True,
    mode="2R",
    negative_slope=0.2,
    dim=2,
):
    assert len(mode) < 4 and mode[0] in [
        "2",
        "3",
    ], "mode examples: 2, 2R, 2BR, 3, ..., 3BR."
    kernel_size_pool = int(mode[0])
    stride_pool = int(mode[0])
    mode = mode.replace(mode[0], "MC")
    pool = conv(
        kernel_size=kernel_size_pool,
        stride=stride_pool,
        mode=mode[0],
        negative_slope=negative_slope,
        dim=dim,
    )
    pool_tail = conv(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        bias,
        mode=mode[1:],
        negative_slope=negative_slope,
        dim=dim,
    )
    return sequential(pool, pool_tail)


# --------------------------------------------
# averagepooling + conv (+ relu)
# --------------------------------------------
def downsample_avgpool(
    in_channels=64,
    out_channels=64,
    kernel_size=3,
    stride=1,
    padding=1,
    bias=True,
    mode="2R",
    negative_slope=0.2,
    dim=2,
):
    assert len(mode) < 4 and mode[0] in [
        "2",
        "3",
    ], "mode examples: 2, 2R, 2BR, 3, ..., 3BR."
    kernel_size_pool = int(mode[0])
    stride_pool = int(mode[0])
    mode = mode.replace(mode[0], "AC")
    pool = conv(
        kernel_size=kernel_size_pool,
        stride=stride_pool,
        mode=mode[0],
        negative_slope=negative_slope,
        dim=dim,
    )
    pool_tail = conv(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        bias,
        mode=mode[1:],
        negative_slope=negative_slope,
        dim=dim,
    )
    return sequential(pool, pool_tail)


def weights_init_drunet(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.orthogonal_(m.weight.data, gain=0.2)


def update_keyvals(old_ckpt, new_ckpt):
    """
    Converting old DRUNet keys to new UNExt style keys.

    KEYS do not change but weight need to be 0 padded.

    Args:
        old_key (str): The old key from the state dictionary.
    """
    # Loop through old state dict and update keys
    # new_ckpt = {}
    for old_key, old_value in old_ckpt.items():
        new_value = new_ckpt[old_key]
        if "head" not in old_key and "tail" not in old_key:
            for _ in range(new_value.shape[-1]):
                new_value[..., _] = old_value / new_value.shape[-1]
            new_ckpt[old_key] = new_value
        else:
            if "head" in old_key:
                c_in = new_value.shape[1]
                c_in_old = old_value.shape[1]
                if c_in == c_in_old:
                    new_value = old_value.detach()
                elif c_in < c_in_old:
                    new_value = torch.zeros_like(new_value.detach())
                    for _ in range(new_value.shape[-1]):
                        new_value[:, -1:, ..., _] = (
                            old_value[:, -1:, ...] / new_value.shape[-1]
                        )
                    for _ in range(new_value.shape[-1]):
                        new_value[:, : c_in - 1, ..., _] = (
                            old_value[:, : c_in - 1, ...] / new_value.shape[-1]
                        )
            elif "tail" in old_key:
                c_in = new_value.shape[0]
                c_in_old = old_value.shape[0]
                new_value = torch.zeros_like(new_value.detach())
                if c_in == c_in_old:
                    new_value = old_value.detach()
                elif c_in < c_in_old:
                    new_value = torch.zeros_like(new_value.detach())
                    for _ in range(new_value.shape[-1]):
                        new_value[:1, ..., _] = old_value[:1, ...] / new_value.shape[-1]
                    for _ in range(new_value.shape[-1]):
                        new_value[1:, ..., _] = (
                            old_value[1:c_in, ...] / new_value.shape[-1]
                        )
            new_ckpt[old_key] = new_value
    return new_ckpt


def test_pad(model, L, modulo=16, sigma: float = 0.0):
    """
    Pads the image to fit the model's expected image size.

    Code borrowed from Kai Zhang https://github.com/cszn/DPIR/tree/master/models
    """
    d, h, w = L.size()[-3:]
    padding_bottom = int(np.ceil(h / modulo) * modulo - h)
    padding_right = int(np.ceil(w / modulo) * modulo - w)
    padding_depth = int(np.ceil(d / modulo) * modulo - d)
    L = torch.nn.ReplicationPad3d(
        (0, padding_depth, 0, padding_right, 0, padding_bottom)
    )(L)
    E = model(L, sigma)
    E = E[..., :d, :h, :w]
    return E


class ComplexDenoiser(torch.nn.Module):
    """Denoiser to wrap DRUNET for Complex data, In 3D."""

    def __init__(self, denoiser, norm_factor=1):
        super().__init__()
        self.denoiser = denoiser
        self.norm_factor = norm_factor

    def forward(self, x, sigma, norm=True):
        x = torch.permute(torch.view_as_real(x.squeeze(0)), (0, 4, 1, 2, 3)).to(
            x.device
        )
        x = x * self.norm_factor
        x_ = torch.permute(
            test_pad(self.denoiser, L=x, sigma=sigma).to(x.device), (0, 2, 3, 4, 1)
        )
        x_ = x_ / self.norm_factor
        return torch.view_as_complex(x_.contiguous()).unsqueeze(0)
