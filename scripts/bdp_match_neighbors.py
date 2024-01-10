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
from bradiphopy.bdp_ops import match_neighbors
from bradiphopy.bradipho_helper import BraDiPhoHelper3D


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('src_files', nargs='+',
                   help='Source (annot) filename (must be supported by VTK).')
    p.add_argument('tgt_file',
                   help='Target filename (must be supported by VTK).')

    p.add_argument('--out_suffix', default='_matched',
                   help='Output suffix (must be supported by VTK) [%(default)s].')
    p.add_argument('--out_dir', default='./',
                   help='Output suffix (must be supported by VTK) [%(default)s].')

    p.add_argument('--distance', type=float, default=0.005,
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

    for i, src_file in enumerate(args.src_files):
        basename = os.path.basename(src_file)
        out_filename = os.path.splitext(basename)[0] + args.out_suffix + '.ply'

        if os.path.isfile(out_filename) and not args.overwrite:
            raise IOError(
                '{} already exists, use -f to overwrite.'.format(args.out_file))

    tgt_polydata = load_polydata(args.tgt_file)
    tgt_bdp_obj = BraDiPhoHelper3D(tgt_polydata)
    annotations = np.zeros((len(tgt_bdp_obj),), dtype=np.uint8)
    distances = np.ones((len(tgt_bdp_obj),), dtype=np.float32) * np.inf

    for i, src_file in enumerate(args.src_files):
        src_polydata = load_polydata(src_file)
        src_bdp_obj = BraDiPhoHelper3D(src_polydata)

        _, curr_indices, curr_distances = match_neighbors(src_bdp_obj,
                                                          tgt_bdp_obj,
                                                          max_dist=args.distance,
                                                          return_indices=True,
                                                          return_distances=True)
        print()
        # Where distances are smaller, replace the annotation
        # and the distance
        replace_indices = curr_distances < distances[curr_indices]
        real_indices = curr_indices[replace_indices]

        annotations[real_indices] = i+1
        distances[real_indices]  = curr_distances[replace_indices]

    for i, src_file in enumerate(args.src_files):
        final_indices = np.where(annotations == i+1)[0]
        basename = os.path.basename(src_file)
        out_filename = os.path.splitext(basename)[0] + args.out_suffix + '.ply'
        tmp_bdp_obj = tgt_bdp_obj.subsample_polydata_vertices(final_indices)
        save_polydata(tmp_bdp_obj.get_polydata(),
                      out_filename, ascii=args.ascii)


if __name__ == '__main__':
    main()
