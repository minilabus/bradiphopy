#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert PolyData from and to any of these extensions:
    [".vtk", ".vtp", ".ply", ".stl", ".xml", ".obj"]

Only PLY file will have visible streamlines in CloudCompare.
MI-Brain does not support coloring of surfaces (when loading).
"""

import argparse
import os

from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.io.stateful_tractogram import (set_sft_logger_level,
                                         StatefulTractogram, Space)
import nibabel as nib
import numpy as np
import vtk

from bradiphopy.io import load_polydata, save_polydata
from bradiphopy.fury import lines_to_vtk_polydata, get_polydata_lines


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_file',
                   help='Input filename (must be supported by VTK).')
    p.add_argument('out_file',
                   help='Output filename (must be supported by VTK).')

    p.add_argument('--scaling', type=float, default=0.001,
                   help='Scaling factor to apply to the streamlines [%(default)s].')
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

    _, ext = os.path.splitext(args.in_file)
    fake_ref = nib.Nifti1Image(np.zeros((10, 10, 10)), np.eye(4))

    if ext in ['.trk', '.tck']:
        ref = args.in_file if ext == '.trk' else fake_ref
        sft = load_tractogram(args.in_file, ref,
                              bbox_valid_check=False)
        sft.streamlines._data = sft.streamlines._data
        polydata = lines_to_vtk_polydata(sft.streamlines)
        # save_tractogram(sft, 'tmp.vtk',
        #                 bbox_valid_check=False)
        # args.in_file = 'tmp.vtk'
    else:
        polydata = load_polydata(args.in_file)
    transform = vtk.vtkTransform()
    transform.Scale([args.scaling]*3)

    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetInputData(polydata)
    transformFilter.SetTransform(transform)
    transformFilter.Update()

    _, ext = os.path.splitext(args.out_file)
    if ext in ['.trk', '.tck']:
        streamlines = get_polydata_lines(transformFilter.GetOutput())
        set_sft_logger_level('DEBUG')

        ref = ref if ext == '.trk' else fake_ref
        sft = StatefulTractogram(streamlines, ref, Space.RASMM)
        save_tractogram(sft, args.out_file,
                        bbox_valid_check=False)
    else:
        save_polydata(transformFilter.GetOutput(), args.out_file,
                      ascii=args.ascii)
    if os.path.isfile('tmp.vtk'):
        os.remove('tmp.vtk')


if __name__ == '__main__':
    main()
