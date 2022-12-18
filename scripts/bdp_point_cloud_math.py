#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Performs an operation on a list of point files. The supported
operations are:

difference:  Keep the points from the first file that are not in
                any of the following files.

intersection: Keep the points that are present in all files.

union:        Keep all points while removing duplicates.

It is assumed that the points are in the same coordinate system.
If the PLY has colors the metadata is preserved in the output file.
"""

import argparse
import os


from bradiphopy.io import load_polydata, save_polydata
from bradiphopy.bradipho_helper import BraDiPhoHelper3D
from bradiphopy.bdp_ops import (apply_intersection,
                                apply_union,
                                apply_difference)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=__doc__)

    p.add_argument('operation', metavar='OPERATION',
                   choices=['intersection', 'union', 'difference'],
                   help='The type of operation to be performed on the '
                        'points. Must\nbe one of the following: '
                        '%(choices)s.')
    p.add_argument('in_files', metavar='INPUT_FILES', nargs='+',
                   help='The list of files that contain the ' +
                        'points to operate on.')
    p.add_argument('out_file', metavar='OUTPUT_FILE',
                   help='The file where the remaining points '
                        'are saved.')

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

    bdp_list = []
    for filename in args.in_files:
        polydata = load_polydata(filename)
        bdp_list.append(BraDiPhoHelper3D(polydata))

    if args.operation == 'intersection':
        bdp_out = apply_intersection(bdp_list)
    elif args.operation == 'union':
        bdp_out = apply_union(bdp_list)
    elif args.operation == 'difference':
        bdp_out = apply_difference(bdp_list)

    save_polydata(bdp_out.get_polydata(), args.out_file, ascii=args.ascii)


if __name__ == "__main__":
    main()
