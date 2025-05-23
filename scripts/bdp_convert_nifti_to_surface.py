#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert NIFTI to surface, by default in RAS coordinates for MI-Brain.
To get it ready FOR CloudCompare, use: --scaling 0.001 and --to_lps
"""

import math
import shutil
import argparse
import os

import numpy as np
import vtk

from bradiphopy.bradipho_helper import BraDiPhoHelper3D
from bradiphopy.io import save_polydata
from bradiphopy.utils import create_mesh_from_image, sample_mesh_to_point_cloud
import nibabel as nib


def _build_arg_parser():
    """Builds and returns an argparse.ArgumentParser for this script.

    The parser is configured with arguments for:
    - Input NIFTI file (.nii or .nii.gz).
    - Output surface file (.ply).
    - Scaling factor for the output surface.
    - A flag to convert coordinates to LPS (Left Posterior Superior).
    - A flag to save as a point cloud instead of a mesh.
    - Sampling distance if saving as a point cloud.
    - Number of dilation iterations for the initial mask.
    - A flag for saving the output file in ASCII format.
    - An overwrite flag for existing output files.
    The script's module-level docstring is used as the description.

    Returns:
        argparse.ArgumentParser: The configured argument parser.
    """
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_file',
                   help='Input filename (.nii or .nii.gz).')
    p.add_argument('out_file',
                   help='Output filename (.ply).')

    p.add_argument('--scaling', type=float, default=1.0,
                   help='Scaling factor to apply to the surface [%(default)s].')
    p.add_argument('--to_lps', action='store_true',
                   help='Flip for Surfice/MI-Brain LPS')
    p.add_argument('--save_point_cloud', action='store_true',
                   help='Save a point cloud instead of a mesh.')
    p.add_argument('--sampling_distance', type=int, default=0.0001,
                   help='Distance between point for sampling the mesh '
                        '[%(default)s].')
    p.add_argument('--dilate', type=int, default=0,
                   help='Number of dilation iterations [%(default)s].')

    p.add_argument('--ascii', action='store_true',
                   help='Save the file with data as ASCII '
                        '(instead of binary).')
    p.add_argument('-f', dest='overwrite', action='store_true',
                   help='Force overwriting of the output files.')
    return p


def main():
    """Main function to convert a NIFTI image to a surface representation.

    Parses command-line arguments. Loads the input NIFTI image.
    Generates an initial mesh from the image data using `create_mesh_from_image`,
    optionally applying dilation.
    Applies user-defined scaling and, if requested, a transformation to LPS
    coordinates.
    If the `--save_point_cloud` option is selected, it samples points from
    the transformed mesh using `sample_mesh_to_point_cloud` and saves this
    point cloud. Otherwise, it saves the transformed mesh.
    The output is a .ply file, optionally in ASCII format.
    """
    parser = _build_arg_parser()
    args = parser.parse_args()

    img = nib.load(args.in_file)
    mesh_poly = create_mesh_from_image(img, dilate=args.dilate)

    obj_point_cloud = BraDiPhoHelper3D(mesh_poly)

    polydata = obj_point_cloud.get_polydata()
    transform = vtk.vtkTransform()
    transform.Scale([args.scaling]*3)
    if args.to_lps:
        flip_LPS = vtk.vtkMatrix4x4()
        flip_LPS.Identity()
        flip_LPS.SetElement(0, 0, -1)
        flip_LPS.SetElement(1, 1, -1)
        transform.Concatenate(flip_LPS)

    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetInputData(polydata)
    transformFilter.SetTransform(transform)
    transformFilter.Update()

    if args.save_point_cloud:
        vertices = sample_mesh_to_point_cloud(transformFilter.GetOutput(),
                                              args.sampling_distance)
        obj_point_cloud = BraDiPhoHelper3D.generate_bdp_obj(vertices)
        save_polydata(obj_point_cloud.get_polydata(), args.out_file,
                      ascii=args.ascii)

    else:
        save_polydata(transformFilter.GetOutput(), args.out_file,
                      ascii=args.ascii)


if __name__ == '__main__':
    main()
