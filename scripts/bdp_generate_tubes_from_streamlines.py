#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to generate tube from TRK files.
"""


import argparse
import os

from dipy.io.streamline import load_tractogram, StatefulTractogram
from dipy.tracking.streamlinespeed import compress_streamlines
import numpy as np
import vtk

from bradiphopy.bradipho_helper import BraDiPhoHelper3D
from bradiphopy.io import save_polydata

from bradiphopy.utils import lines_to_vtk_polydata


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_bundle',
                   help='Requires the filename and radius (mm) as a pair.')
    p.add_argument('radius', type=float,
                   help='Radius of the tubes.')
    p.add_argument('out_file',
                   help='Output bundle filename.')
    p.add_argument('--color', nargs=3, type=int, default=(255, 255, 255),
                   help='Color as RGB (0-255).')
    p.add_argument('--ascii', action='store_true',
                   help='Save the file with data as ASCII '
                        '(instead of binary).')
    p.add_argument('-f', dest='overwrite', action='store_true',
                   help='Force overwriting of the output files.')
    return p


def create_tubes(streamlines, radius):
    polydata = lines_to_vtk_polydata(streamlines)

    # Tube filter for the rendering with varying radii
    tubeFilter = vtk.vtkTubeFilter()
    tubeFilter.SetInputData(polydata)
    tubeFilter.SetRadius(float(radius))
    tubeFilter.SetNumberOfSides(6)
    tubeFilter.CappingOn()
    tubeFilter.Update()

    return tubeFilter.GetOutput()


def compress_sft(sft, tol_error=0.2):
    """ """
    # Go to world space
    orig_space = sft.space
    sft.to_rasmm()

    compressed_streamlines = compress_streamlines(sft.streamlines,
                                                  tol_error=tol_error)

    compressed_sft = StatefulTractogram.from_sft(
        compressed_streamlines, sft,
        data_per_streamline=sft.data_per_streamline)

    # Return to original space
    compressed_sft.to_space(orig_space)

    return compressed_sft


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if os.path.isfile(args.out_file) and not args.overwrite:
        raise IOError(
            '{} already exists, use -f to overwrite.'.format(args.out_file))

    aff = np.eye(3)
    aff[0, 0] = -1
    aff[1, 1] = -1

    sft = compress_sft(load_tractogram(args.in_bundle, 'same'))
    sft.streamlines._data = np.dot(sft.streamlines._data, aff)
    polydata = create_tubes(sft.streamlines, args.radius)
    obj_w_strip = BraDiPhoHelper3D(polydata)

    obj_w_polys = BraDiPhoHelper3D.generate_bdp_obj(
        obj_w_strip.get_polydata_vertices(),
        obj_w_strip.get_polydata_triangles())

    # Flip triangles
    normal_generator = vtk.vtkPolyDataNormals()
    normal_generator.SetInputData(obj_w_polys.get_polydata())
    normal_generator.ComputeCellNormalsOff()
    normal_generator.ConsistencyOn()
    normal_generator.Update()

    obj_out = BraDiPhoHelper3D(normal_generator.GetOutput())
    color_arr = np.zeros((len(obj_out), 3))
    color_arr[:] = args.color
    obj_out.set_scalar(color_arr, 'RGB', dtype='uint8')

    save_polydata(obj_out.get_polydata(), args.out_file, ascii=args.ascii)


if __name__ == '__main__':
    main()