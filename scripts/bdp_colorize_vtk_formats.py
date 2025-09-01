#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Colorize PolyData from and to any of these extensions:
    [".vtk", ".vtp", ".ply", ".stl", ".xml", ".obj"]

In the context of the Bradipho project, most scripts expect the PLY format.

The script will colorize by modifying the Hue of the HSV color space.
The saturation and value can be modified with the --saturation_target,
--saturation_multiplier, --value_target, and --value_multiplier options.
(this is to control the obtained color, since every scene/epoch is different)
"""

import argparse
import colorsys
import json
import os

import matplotlib
import numpy as np

from bradiphopy.bradipho_helper import BraDiPhoHelper3D
from bradiphopy.io import load_polydata, save_polydata
from bradiphopy.utils import get_colormap


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument("in_file", help="Input filename (must be supported by VTK).")
    p.add_argument("out_file", help="Output filename (must be supported by VTK).")
    p2 = p.add_mutually_exclusive_group(required=True)
    p2.add_argument("--color", nargs=3, type=int, help="Color as RGB (0-255).")
    p2.add_argument(
        "--colormap",
        nargs="?",
        const="jet",
        help="Select the colormap for axis ordering coloring "
        "[%(default)s].\nUse two Matplotlib named color "
        "separeted by a - to create your own colormap.",
    )
    p2.add_argument(
        "--cLUT",
        help="Select the colormap from a .json file containing a "
        "color LUT (uses basename as key in the dict).",
    )
    p.add_argument(
        "--axis",
        default="auto",
        choices=["x", "y", "z", "auto"],
        help="Axis to use for the colormap [%(choices)s].",
    )

    p.add_argument_group("Color manipulation")
    s = p.add_mutually_exclusive_group()
    s.add_argument(
        "--saturation_target",
        type=float,
        const=0.6,
        nargs="?",
        help="Target saturation. Must be lower than 1.0\n"
        "Lower = Move toward white, gray, black [%(default)s].",
    )
    s.add_argument(
        "--saturation_multiplier",
        type=float,
        const=1.0,
        nargs="?",
        help="Saturation multiplier. Must be lower than 1.0\n"
        "Lower = Move toward white, gray, black [%(default)s].",
    )
    v = p.add_mutually_exclusive_group()
    v.add_argument(
        "--value_target",
        type=float,
        const=160,
        nargs="?",
        help="Target value. Must be lower than 255\n"
        "Should be around 100-192 [%(default)s].",
    )
    v.add_argument(
        "--value_multiplier",
        type=float,
        const=1.8,
        nargs="?",
        help="Value multiplier. Higher = More intense color\n"
        "Should be around 1.25-2.0 [%(default)s].",
    )
    p.add_argument(
        "--ascii",
        action="store_true",
        help="Save the file with data as ASCII (instead of binary).",
    )
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

    if os.path.isfile(args.out_file) and not args.overwrite:
        raise IOError(
            "{} already exists, use -f to overwrite.".format(args.out_file)
        )

    _, ext = os.path.splitext(args.out_file)

    polydata = load_polydata(args.in_file)
    bdp_obj = BraDiPhoHelper3D(polydata)
    vertices = bdp_obj.get_polydata_vertices()

    basename = os.path.basename(args.in_file)
    if args.cLUT:
        with open(args.cLUT, "r") as f:
            cLUT = json.load(f)
        name = os.path.splitext(basename)[0]
        if name in cLUT:
            args.color = [c for c in cLUT[name]]
        else:
            args.color = [0, 0, 0]

    try:
        rgb_scalar = bdp_obj.get_scalar("RGB")
        empty = False
    except ValueError:
        rgb_scalar = np.ones(vertices.shape) * 255
        empty = True

    hsv_scalar = matplotlib.colors.rgb_to_hsv(rgb_scalar)

    if args.color:
        if empty:
            rgb_scalar = np.empty_like(vertices)
            rgb_scalar[:] = args.color
        else:
            hsv_scalar[:, 0] = colorsys.rgb_to_hsv(*args.color)[0]
            hsv_scalar[:, 1] = colorsys.rgb_to_hsv(*args.color)[1]
            rgb_scalar = matplotlib.colors.hsv_to_rgb(hsv_scalar)
    else:
        bbox = np.min(vertices, axis=0), np.max(vertices, axis=0)
        if args.axis == "auto":
            axis = np.argmax(np.abs(bbox[0] - bbox[1]))
        else:
            axis = "xyz".index(args.axis)

        cmap = get_colormap(args.colormap)

        normalized_value = (vertices[:, axis] - bbox[0][axis]) / (
            bbox[1][axis] - bbox[0][axis]
        )
        rgb_scalar = cmap(normalized_value)[:, 0:3]

        if empty:
            rgb_scalar *= 255

    hsv_scalar[:, 0] = matplotlib.colors.rgb_to_hsv(rgb_scalar)[:, 0]

    if args.saturation_target is not None:
        hsv_scalar[:, 1] *= args.saturation_target / hsv_scalar[:, 1].mean()
    if args.value_target is not None:
        hsv_scalar[:, 2] *= args.value_target / hsv_scalar[:, 2].mean()
    if args.saturation_multiplier is not None:
        hsv_scalar[:, 1] *= args.saturation_multiplier
    if args.value_multiplier is not None:
        hsv_scalar[:, 2] *= args.value_multiplier

    np.clip(hsv_scalar[:, 0], 0, 1, out=hsv_scalar[:, 0])
    np.clip(hsv_scalar[:, 1], 0, 1, out=hsv_scalar[:, 1])
    np.clip(hsv_scalar[:, 2], 0, 255, out=hsv_scalar[:, 2])
    rgb_scalar = matplotlib.colors.hsv_to_rgb(hsv_scalar)

    bdp_obj.set_scalar(rgb_scalar, "RGB", dtype="uint8")
    save_polydata(bdp_obj.get_polydata(), args.out_file, ascii=args.ascii)


if __name__ == "__main__":
    main()
