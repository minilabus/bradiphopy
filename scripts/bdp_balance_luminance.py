#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Harmonize the luminance of a set of images from photometric scene.
Will balance the luminance of the images to the average luminance of the set
using a simple linear transformation of the HSV space.
"""

import argparse
import os
import shutil

import imageio.v2 as imageio
import matplotlib
import numpy as np


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument("in_files", nargs="+", help="Input filenames (.jpg or .png).")
    p.add_argument("out_dir", help="Output directory for balanced images.")
    p.add_argument(
        "-f",
        dest="overwrite",
        action="store_true",
        help="Force overwriting of the output files.",
    )
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if os.path.isdir(args.out_dir):
        if not args.overwrite:
            raise IOError(
                "{} already exists, use -f to overwrite.".format(args.out_dir)
            )
        else:
            shutil.rmtree(args.out_dir)
    os.makedirs(args.out_dir)

    img_rgb_list = []
    for in_file in args.in_files:
        img = imageio.imread(in_file)
        img_rgb_list.append(img)

    img_hsv_list = []
    for rgb_img in img_rgb_list:
        img = matplotlib.colors.rgb_to_hsv(rgb_img)
        img_hsv_list.append(img)
    img_hsv_arr = np.stack(img_hsv_list, axis=-1)

    avg_val = np.average(img_hsv_arr, axis=(0, 1, -1))
    min_val = np.min(img_hsv_arr, axis=(0, 1, -1))
    max_val = np.max(img_hsv_arr, axis=(0, 1, -1))

    for i, hsv_img in enumerate(img_hsv_list):
        curr_min = np.min(hsv_img, axis=(0, 1))

        hsv_img -= curr_min + min_val
        curr_avg = np.average(hsv_img, axis=(0, 1))
        hsv_img *= avg_val / curr_avg

        hsv_img = np.clip(hsv_img, 0, max_val)
        rgb_img = matplotlib.colors.hsv_to_rgb(hsv_img)

        basename = os.path.basename(args.in_files[i])
        out_file = os.path.join(args.out_dir, basename)
        imageio.imwrite(out_file, rgb_img.astype(np.uint8))


if __name__ == "__main__":
    main()
