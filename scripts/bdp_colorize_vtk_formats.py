#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert PolyData from and to any of these extensions:
    [".vtk", ".vtp", ".ply", ".stl", ".xml", ".obj"]
"""

import argparse
import colorsys
import os

import matplotlib
import numpy as np

from bradiphopy.io import load_polydata, save_polydata
from bradiphopy.bradipho_helper import BraDiPhoHelper3D


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_surface',
                   help='Input filename (must be supported by VTK.')
    p.add_argument('out_surface',
                   help='Output filename (must be supported by VTK.')
    p.add_argument('--colors', nargs=3, type=int, required=True,
                   help='Input filename (must be supported by VTK.')

    p.add_argument('--binary', action='store_true',
                   help='Save the file with data as raw binary '
                        '(instead of ASCII).')
    p.add_argument('-f', dest='overwrite', action='store_true',
                   help='Force overwriting of the output files.')
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if os.path.isfile(args.out_surface) and not args.overwrite:
        raise IOError(
            '{} already exists, use -f to overwrite.'.format(args.out_surface))

    polydata = load_polydata(args.in_surface)
    bdp_obj = BraDiPhoHelper3D(polydata)
    rgb_scalar = bdp_obj.get_scalar('RGB')

    hsv_scalar = matplotlib.colors.rgb_to_hsv(rgb_scalar)
    hsv_scalar[:, 0] = colorsys.rgb_to_hsv(*args.colors)[0]
    rgb_scalar = matplotlib.colors.hsv_to_rgb(hsv_scalar)

    bdp_obj.set_scalar(rgb_scalar, 'RGB', dtype=np.uint8)
    save_polydata(polydata, args.out_surface, binary=args.binary)


if __name__ == '__main__':
    main()
