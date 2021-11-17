from pathlib import Path
import warnings

import cv2
import numpy as np
from PIL import Image
import torch

from .colormap import voc_colormap


class ToneLabel:
    def __init__(self, label: np.ndarray, ignore: set = {0}) -> None:
        assert label.dtype == np.uint8
        self.data = label
        self.ignore = ignore

    @classmethod
    def load(cls, path: str, dtype=np.uint8):
        data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return cls(data.astype(dtype))

    @property
    def shape(self) -> tuple:
        return self.data.shape

    def save(self, path: str):
        return cv2.imwrite(path, self.data)

    def get_tensor(self) -> torch.Tensor:
        return torch.from_numpy(self.data.astype(np.int32))

    def visualize(self) -> Image.Image:
        colors = voc_colormap(range(200)).astype(np.uint8)
        return Image.fromarray(colors[self.data])


class ToneParameter:
    def __init__(self, tone_index: int, tone_type: str, mask, param: dict) -> None:
        self.tone_index = tone_index
        self.tone_type = tone_type
        self.mask = mask
        self.param = param

    def __repr__(self) -> str:
        return (
            self.__class__.__name__
            + "(tone_index={}, tone_type={}, param={})".format(
                self.tone_index, self.tone_type, self.param
            )
        )


class ToneImageGenerator:
    def __init__(self, data_root: str = "./data/") -> None:
        self.tone_dataset = ToneDataset(data_root)

    def render(self, label: ToneLabel, line=None) -> np.ndarray:
        params = self.label_to_params(label)
        if line is None:
            # prepare a white image
            shape = params[0]
            img = np.ones(shape, dtype=np.uint8) * 255
        else:
            img = line.copy()

        for tone_param in params[1:]:
            if tone_param.tone_type == "unlabeled":
                continue
            tone = self.generate(tone_param)
            img = np.minimum(tone, img)

        return img.astype(np.uint8)

    def label_index_to_param(self, lb: int) -> ToneParameter:
        tone_index = lb - 1
        for tone_type in ToneDataset.tone_types:
            if 0 <= tone_index < len(self.tone_dataset.data[tone_type]):
                break
            tone_index -= len(self.tone_dataset.data[tone_type])

        else:
            tone_type = "unlabeled"
            assert tone_index == 0

        param = dict()
        if tone_type in {"gradation", "dark"}:
            pass
        elif tone_type == "effect":
            param["scale"] = 1.0
        elif tone_type == "light":
            param["scale_inv"] = 1.0
            param["angle"] = 0

        param["value_scale"] = 1.0
        tone_param = ToneParameter(tone_index, tone_type, None, param)

        return tone_param

    def label_to_params(self, label: ToneLabel) -> list:
        params = list()
        params.append(label.shape)

        # prepare label_set that renders
        label_set = set(np.unique(label.data))
        label_set -= label.ignore

        for lb in label_set:
            mask = label.data == lb
            tone_param = self.label_index_to_param(lb)
            tone_param.mask = mask * 255.0
            params.append(tone_param)

        return params

    def generate(self, tone_param: ToneParameter) -> np.ndarray:
        """generate screentones from tone_param

        modified from the code by Chengze Li.
        """
        tile = self.tone_dataset.get(tone_param.tone_index, tone_param.tone_type)
        mask = tone_param.mask

        if tone_param.tone_type == "gradation":
            result = np.ones(mask.shape, np.float32) * 255.0
            h_tile = tile.shape[0]

            mask_bin = mask == 255.0
            xmin, xmax = np.where(np.any(mask_bin, axis=0))[0][[0, -1]]
            ymin, ymax = np.where(np.any(mask_bin, axis=1))[0][[0, -1]]

            h_box, w_box = ymax - ymin, xmax - xmin
            # NOTE: (h_box: height of a contour rectangular) + 1 <= (h_tile: height of a tone image)
            if h_tile >= h_box:
                crop = tile[0:h_box, 0:w_box]
                result[ymin:ymax, xmin:xmax] = crop
            else:
                warnings.warn(
                    "Unexpected label. Unable to paste gradation.", RuntimeWarning
                )

        elif tone_param.tone_type == "effect":
            # height, width for resize
            height_t, width_t = tile.shape
            height, width = mask.shape

            scaler = height / float(height_t)
            scalec = width / float(width_t)

            scale = (max(scaler, scalec) + 1) / 2

            height_t = max(height, int(height_t * scale) + 1)
            width_t = max(width, int(width_t * scale) + 1)

            newtile = np.ones((height_t + 1, width_t + 1), np.float32) * 255.0
            effect = cv2.resize(tile, (width_t, height_t), interpolation=cv2.INTER_AREA)
            newtile[0:height_t, 0:width_t] = effect

            # center crop
            rr = (0 + height_t - height + 1) // 2
            rc = (0 + width_t - width + 1) // 2
            result = newtile[rr : rr + height, rc : rc + width]

        elif tone_param.tone_type == "dark":
            height, width = mask.shape
            result = cv2.resize(tile, (width, height), interpolation=cv2.INTER_AREA)

        elif tone_param.tone_type == "light":
            scale_inv = tone_param.param["scale_inv"]
            angle = tone_param.param["angle"]

            height, width = mask.shape

            assert scale_inv == 1.0 and (not angle)
            rowtiles = height // tile.shape[0] + 1
            coltiles = width // tile.shape[1] + 1
            tile_dest = np.tile(tile, (rowtiles, coltiles))
            result = tile_dest[:height, :width]

        else:
            raise NotImplementedError

        # edit
        value_scale = tone_param.param["value_scale"]
        tile = result * value_scale
        tile[tile > 255] = 255

        # apply mask
        tile = mask * tile / 255.0 + (255 - mask)
        return tile


class ToneDataset:

    tone_types = ("gradation", "effect", "light", "dark")

    def __init__(self, root: str, grayscale: bool = True) -> None:
        self.root = Path(root)
        data_root = {
            "light": "screenPatterns/light/",
            "effect": "secretsanta2011/effects/",
            "gradation": "secretsanta2011/gradations/",
        }
        self.data = dict()
        for tone_type in self.tone_types:
            if tone_type == "dark":
                self.data[tone_type] = [np.zeros((10, 10), dtype=np.uint8)]
            else:
                if tone_type in {"light", "effect"}:
                    paths = sorted((self.root / data_root[tone_type]).glob("*"))
                else:
                    paths = sorted((self.root / data_root[tone_type]).glob("*/*/*"))
                color = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_UNCHANGED
                self.data[tone_type] = [
                    cv2.imread(path.as_posix(), color) for path in paths
                ]

    def __len__(self) -> int:
        return sum(len(self.data[tone_type]) for tone_type in self.tone_types)

    def get(self, index: int, tone_type) -> np.ndarray:
        return self.data[tone_type][index]
