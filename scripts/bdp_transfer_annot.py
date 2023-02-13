#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Match neighbors from one point_cloud to another.
Works better the closer the two files are in space.
"""

import argparse
import os
import numpy as np

from bradiphopy.io import load_polydata, save_polydata
from bradiphopy.bdp_ops import transfer_annots
from bradiphopy.bradipho_helper import BraDiPhoHelper3D


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('src_file', nargs='+',
                   help='Source (annot) filenames (must be supported by VTK).')
    p.add_argument('tgt_file',
                   help='Target filename (must be supported by VTK).')
    p.add_argument('out_file',
                   help='Output filename (must be supported by VTK).')
    p.add_argument('--annot_lut',
                   help='LUT for consistency between Epoch/Subject')
    p.add_argument('--distance', type=float, default=0.002,
                   help='Maximum distance for transfer (mm) [%(default)s]')
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

    src_polydata = [load_polydata(filename) for filename in args.src_file]
    tgt_polydata = load_polydata(args.tgt_file)

    src_bdp_obj = [BraDiPhoHelper3D(polydata) for polydata in src_polydata]
    tgt_bdp_obj = BraDiPhoHelper3D(tgt_polydata)

    _, new_annots = transfer_annots(src_bdp_obj, tgt_bdp_obj,
                                    distance=args.distance,
                                    filenames=args.src_file,
                                    annot_lut=args.annot_lut)
    np.savetxt(args.out_file, new_annots, fmt='%i')


if __name__ == '__main__':
    main()
