#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert surface to NIFTI file.
Must be aligned with the reference NIFTI file in MI-Brain.
To get it aligned FROM CloudCompare, use: --scaling 1000 and --to_lps
"""

import argparse
import os

import nibabel as nib
from nibabel.affines import apply_affine
import numpy as np

from bradiphopy.bradipho_helper import BraDiPhoHelper3D
from bradiphopy.io import load_polydata


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_surface',
                   help='Input surface file.')
    p.add_argument('in_ref',
                   help='Input reference NIFTI file.')
    p.add_argument('out_file',
                   help='Output NIFTI file.')
    p.add_argument('-f', dest='overwrite', action='store_true',
                   help='Force overwriting of the output files.')
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if os.path.exists(args.out_file) and not args.overwrite:
        parser.error('Output file already exists. Use -f to overwrite.')

    ref_img = nib.load(args.in_ref)
    polydata = load_polydata(args.in_surface)
    bdp_obj = BraDiPhoHelper3D(polydata)

    vertices = bdp_obj.get_polydata_vertices()
    inv_affine = np.linalg.inv(ref_img.affine)
    flip_affine = np.diag([-1, -1, 1, 1])
    vertices = apply_affine(flip_affine, vertices)
    vertices = apply_affine(inv_affine, vertices) + 0.5
    vertices = vertices.astype(int)

    density = np.zeros(ref_img.shape)
    density[vertices[:, 0], vertices[:, 1], vertices[:, 2]] = 1

    nib.save(nib.Nifti1Image(density, ref_img.affine), args.out_file)


if __name__ == '__main__':
    main()
