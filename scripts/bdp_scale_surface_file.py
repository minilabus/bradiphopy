#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert PolyData from and to any of these extensions:
    [".vtk", ".vtp", ".ply", ".stl", ".xml", ".obj"]

Useful to switch between space CloudCompare and MI-Brain.
CloudCompare is in meters, MI-Brain is in millimeters (so --scaling 1000) will
convert a surface from CloudCompare to MI-Brain (and --to_lps will flip it).
"""

import argparse
import os

from bradiphopy.io import load_polydata, save_polydata
import vtk


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_file',
                   help='Input filename (must be supported by VTK).')
    p.add_argument('out_file',
                   help='Output filename (must be supported by VTK).')

    p.add_argument('--scaling', type=float, default=0.001,
                   help='Scaling factor to apply to the streamlines [%(default)s].')
    p.add_argument('--to_lps', action='store_true',
                   help='Flip for Surfice/MI-Brain LPS')

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

    polydata = load_polydata(args.in_file, to_lps=args.to_lps)
    transform = vtk.vtkTransform()
    transform.Scale([args.scaling]*3)

    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetInputData(polydata)
    transformFilter.SetTransform(transform)
    transformFilter.Update()

    save_polydata(transformFilter.GetOutput(), args.out_file, ascii=args.ascii)


if __name__ == '__main__':
    main()
