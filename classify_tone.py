import argparse

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms

from utils.screentone import ToneLabel


class ResBlock(nn.Module):
    def __init__(self, in_size: int, out_size: int, mid_size=None) -> None:
        super(ResBlock, self).__init__()
        if mid_size is None:
            mid_size = out_size

        self.layer_1 = nn.Sequential(
            nn.Conv2d(in_size, mid_size, 3, padding=1),
            nn.BatchNorm2d(mid_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_size, out_size, 3, padding=1),
            nn.BatchNorm2d(out_size),
        )
        self.layer_2 = nn.Sequential(
            nn.Conv2d(in_size, out_size, 3, padding=1), nn.BatchNorm2d(out_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.layer_1(x) + self.layer_2(x), inplace=True)


class DownResBlock(nn.Module):
    def __init__(self, in_size: int, out_size: int, mid_size=None) -> None:
        super(DownResBlock, self).__init__()
        if mid_size is None:
            mid_size = out_size

        self.layer_1 = nn.Sequential(
            nn.Conv2d(in_size, mid_size, 3, padding=1, stride=2),
            nn.BatchNorm2d(mid_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_size, out_size, 3, padding=1),
            nn.BatchNorm2d(out_size),
        )
        self.layer_2 = nn.Sequential(
            nn.Conv2d(in_size, out_size, 3, padding=1, stride=2),
            nn.BatchNorm2d(out_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.layer_1(x) + self.layer_2(x), inplace=True)


class UpResBlock(nn.Module):
    def __init__(self, in_size: int, out_size: int, mid_size=None) -> None:
        super(UpResBlock, self).__init__()
        if mid_size is None:
            mid_size = out_size

        self.layer_1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_size, mid_size, 3, padding=1, stride=2, output_padding=1
            ),
            nn.BatchNorm2d(mid_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_size, out_size, 3, padding=1),
            nn.BatchNorm2d(out_size),
        )
        self.layer_2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_size, out_size, 3, padding=1, stride=2, output_padding=1
            ),
            nn.BatchNorm2d(out_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.layer_1(x) + self.layer_2(x), inplace=True)


def init_weight(layer):
    if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
        nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")


class ResidualUNet(nn.Module):
    """Residual U-Net"""

    def __init__(
        self,
        in_size: int = 1,
        n_class: int = 11,
        pretrained_model=None,
        feature_scale=1,
    ) -> None:
        super(ResidualUNet, self).__init__()
        self.n_class = n_class
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        self.filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.cb1 = nn.Sequential(
            ResBlock(in_size, self.filters[0]),
            ResBlock(self.filters[0], self.filters[0]),
        )
        self.cb2 = nn.Sequential(
            DownResBlock(self.filters[0], self.filters[1]),
            ResBlock(self.filters[1], self.filters[1]),
        )
        self.cb3 = nn.Sequential(
            DownResBlock(self.filters[1], self.filters[2]),
            ResBlock(self.filters[2], self.filters[2]),
        )
        self.cb4 = nn.Sequential(
            DownResBlock(self.filters[2], self.filters[3]),
            ResBlock(self.filters[3], self.filters[3]),
        )
        self.cb5 = nn.Sequential(
            DownResBlock(self.filters[3], self.filters[4]),
            ResBlock(self.filters[4], self.filters[4]),
            ResBlock(self.filters[4], self.filters[4]),
        )

        # upsampling
        self.cb6 = nn.Sequential(
            UpResBlock(self.filters[4], self.filters[3]),
            ResBlock(self.filters[3], self.filters[3]),
        )
        self.cb7 = nn.Sequential(
            UpResBlock(self.filters[3], self.filters[2]),
            ResBlock(self.filters[2], self.filters[2]),
        )
        self.cb8 = nn.Sequential(
            UpResBlock(self.filters[2], self.filters[1]),
            ResBlock(self.filters[1], self.filters[1]),
        )
        self.cb9 = nn.Sequential(
            UpResBlock(self.filters[1], self.filters[0]),
            ResBlock(self.filters[0], self.filters[0]),
        )

        self.cb10 = nn.Sequential(
            ResBlock(self.filters[0], self.filters[0]),
        )

        # final conv (without any concat)
        self.conv_classifier = nn.Conv2d(self.filters[0], n_class, 1)

        if pretrained_model:
            self.load_state_dict(torch.load(pretrained_model))

        else:
            self.apply(init_weight)

    def __call__(self, x: torch.Tensor, with_feat=False, output_label=False):

        # (1, 256, 256) -> (64, 256, 256)
        cb1 = self.cb1(x)
        # (64, 256, 256) -> (128, 128, 128)
        cb2 = self.cb2(cb1)
        # (128, 128, 128) -> (256, 64, 64)
        cb3 = self.cb3(cb2)
        # (256, 64, 64) -> (512, 32, 32)
        cb4 = self.cb4(cb3)
        # (512, 32, 32) -> (1024, 16, 16)
        cb5 = self.cb5(cb4)
        cb4 = cb4 + self.cb6(cb5)
        cb3 = cb3 + self.cb7(cb4)
        cb2 = cb2 + self.cb8(cb3)
        cb1 = cb1 + self.cb9(cb2)
        h = self.cb10(cb1)
        y = self.conv_classifier(h)

        if output_label:
            y = torch.argmax(y, dim=1)

        if with_feat:
            return y, h
        else:
            return y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="path for a manga image")
    parser.add_argument(
        "--out", default="label-c.png", help="output path of a screentone label"
    )
    parser.add_argument("--n_class", default=120)
    parser.add_argument("--model_path", default="unet.pth")
    args = parser.parse_args()

    model = ResidualUNet(n_class=args.n_class, pretrained_model=args.model_path)
    model = model.eval()

    with Image.open(args.path) as img_pil:
        W, H = img_pil.size

    H_pad: int = (H + 15) // 16 * 16 - H
    W_pad: int = (W + 15) // 16 * 16 - W
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Pad((0, 0, H_pad, W_pad), fill=255),
            transforms.ToTensor(),
            transforms.Normalize((0.0,), (1 / 255.0,)),
        ]
    )
    img_t: torch.Tensor = transform(img_pil)

    # (1, n_class, H_pad, W_pad) -> (n_class, H_pad, W_pad) -> (H_pad, W_pad) -> (H, W)
    with torch.no_grad():
        label: torch.Tensor = model(img_t[None])[0].argmax(dim=0)[0:H, 0:W]
    tone_label: ToneLabel = ToneLabel(label.numpy().astype(np.uint8))
    tone_label.save(args.out)


if __name__ == "__main__":
    main()
