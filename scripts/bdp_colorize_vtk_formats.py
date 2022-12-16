#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Colorize PolyData from and to any of these extensions:
    [".vtk", ".vtp", ".ply", ".stl", ".xml", ".obj"]
"""

import argparse
import colorsys
import os

import matplotlib

from bradiphopy.bradipho_helper import BraDiPhoHelper3D
from bradiphopy.io import load_polydata, save_polydata


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_file',
                   help='Input filename (must be supported by VTK).')
    p.add_argument('out_file',
                   help='Output filename (must be supported by VTK).')
    p.add_argument('--color', nargs=3, type=int, required=True,
                   help='Color as RGB (0-255).')

    p.add_argument('--ascii', action='store_true',
                   help='Save the file with data as ASCII '
                        '(instead of binary).')
    p.add_argument('-f', dest='overwrite', action='store_true',
                   help='Force overwriting of the output files.')
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if os.path.isfile(args.out_file) and not args.overwrite:
        raise IOError(
            '{} already exists, use -f to overwrite.'.format(args.out_file))

    polydata = load_polydata(args.in_file)
    bdp_obj = BraDiPhoHelper3D(polydata)
    rgb_scalar = bdp_obj.get_scalar('RGB')

    hsv_scalar = matplotlib.colors.rgb_to_hsv(rgb_scalar)
    hsv_scalar[:, 0] = colorsys.rgb_to_hsv(*args.color)[0]
    rgb_scalar = matplotlib.colors.hsv_to_rgb(hsv_scalar)

    bdp_obj.set_scalar(rgb_scalar, 'RGB', dtype='uint8')
    save_polydata(bdp_obj.get_polydata(), args.out_file, ascii=args.ascii)


if __name__ == '__main__':
    main()
