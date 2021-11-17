import argparse

import cv2
import numpy as np
from PIL import Image

from utils.post_proc import uniform_label
from utils.screentone import ToneLabel, ToneImageGenerator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("line", help="line drawing")
    parser.add_argument("label", help="screentone label")
    parser.add_argument("--out", default="out.png")
    parser.add_argument(
        "--out-label",
        default="label-vis.png",
        help="visualization of the screentone label",
    )
    args = parser.parse_args()

    tone_gen = ToneImageGenerator()

    # read a line drawing
    if args.line is None:
        line = None
    else:
        with Image.open(args.line) as f:
            line = f.convert("L").resize((256, 256), Image.BICUBIC)
            line = np.asarray(line, dtype=np.uint8)

    # read a tone label
    label = ToneLabel.load(args.label)
    label.visualize().save(args.out_label)

    # post-process
    label.data = uniform_label(label.data, line, thresh=144)
    # render a manga image
    img_rec = tone_gen.render(label, line)
    # save the manga image
    cv2.imwrite(args.out, img_rec)


if __name__ == "__main__":
    main()
