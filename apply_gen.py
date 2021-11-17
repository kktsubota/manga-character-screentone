import argparse
import functools

import numpy as np
from PIL import Image
import torch
from torch import nn
from torchvision import transforms

from utils.screentone import ToneLabel


class UnetGenerator(nn.Module):
    """Create a Unet-based generator

    we modify the output layer from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    Copyright (c) 2016, Phillip Isola and Jun-Yan Zhu
    """

    def __init__(
        self,
        input_nc,
        output_nc,
        num_downs,
        ngf=64,
        norm_layer=nn.BatchNorm2d,
        use_dropout=False,
        last_act="tanh",
    ):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(
            ngf * 8,
            ngf * 8,
            input_nc=None,
            submodule=None,
            norm_layer=norm_layer,
            innermost=True,
        )  # add the innermost layer
        for i in range(num_downs - 5):  # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(
                ngf * 8,
                ngf * 8,
                input_nc=None,
                submodule=unet_block,
                norm_layer=norm_layer,
                use_dropout=use_dropout,
            )
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(
            ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer
        )
        unet_block = UnetSkipConnectionBlock(
            ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer
        )
        unet_block = UnetSkipConnectionBlock(
            ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer
        )
        self.model = UnetSkipConnectionBlock(
            output_nc,
            ngf,
            input_nc=input_nc,
            submodule=unet_block,
            outermost=True,
            norm_layer=norm_layer,
            last_act=last_act,
        )  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
    X -------------------identity----------------------
    |-- downsampling -- |submodule| -- upsampling --|

    we modify the output layer from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    Copyright (c) 2016, Phillip Isola and Jun-Yan Zhu
    """

    def __init__(
        self,
        outer_nc,
        inner_nc,
        input_nc=None,
        submodule=None,
        outermost=False,
        innermost=False,
        norm_layer=nn.BatchNorm2d,
        use_dropout=False,
        last_act="tanh",
    ):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            user_dropout (bool) -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(
            input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias
        )
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            down = [downconv]

            # original code
            # if last_act == 'tanh':
            #     upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
            #                                 kernel_size=4, stride=2,
            #                                 padding=1)
            #     up = [uprelu, upconv, nn.Tanh()]

            # 64 * 2 => 32
            upconv = nn.ConvTranspose2d(
                inner_nc * 2,
                inner_nc // 2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=use_bias,
            )
            upnorm = norm_layer(inner_nc // 2)
            lastconv = nn.Conv2d(inner_nc // 2, outer_nc, kernel_size=1)
            up = [uprelu, upconv, upnorm, uprelu, lastconv]
            if last_act == "tanh":
                up += [nn.Tanh()]
            elif last_act == "logSoftmax":
                up += [nn.LogSoftmax(dim=1)]
            else:
                raise NotImplementedError

            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(
                inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias
            )
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(
                inner_nc * 2,
                outer_nc,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=use_bias,
            )
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:  # add skip connections
            return torch.cat([x, self.model(x)], 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("line", help="line drawing")
    parser.add_argument("--model_path")
    parser.add_argument(
        "--out", default="label.png", help="output path of a screentone label"
    )
    args = parser.parse_args()

    with Image.open(args.line) as f:
        img = f.convert("L")

    transform = transforms.Compose(
        [
            transforms.Resize((256, 256), transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    img_t = transform(img)

    norm_layer = functools.partial(
        nn.BatchNorm2d, affine=True, track_running_stats=True
    )
    net = UnetGenerator(
        1, 120, 8, 64, norm_layer=norm_layer, use_dropout=True, last_act="logSoftmax"
    )
    if args.model_path is not None:
        state_dict = torch.load(args.model_path)
        net.load_state_dict(state_dict)

    # We do not use eval mode to generate dirverse output.
    # So, the output can differ for each run.
    # net.eval()

    with torch.no_grad():
        out = net(img_t[None])[0]
        label_data = out.argmax(dim=0)

    label = ToneLabel(label_data.numpy().astype(np.uint8))
    label.save(args.out)


if __name__ == "__main__":
    main()
