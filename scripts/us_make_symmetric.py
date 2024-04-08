#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Make a symmetric image from two images (hemispheres).
Utility script for the generate_all_parcellation_and_point_cloud.sh recipe.
"""

import argparse
import os

import nibabel as nib
import numpy as np


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_file_1',
                   help='Input file (nifti), one hemisphere.')
    p.add_argument('in_file_2',
                   help='Input file (nifti), the other hemisphere.')
    p.add_argument('out_file',
                   help='Output file (nifti).')
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if os.path.isfile(args.out_file) and not args.overwrite:
        raise IOError(
            '{} already exists, use -f to overwrite.'.format(args.out_file))

    img_1 = nib.load(args.in_file_1)
    img_2 = nib.load(args.in_file_2)

    # Roll the arrays to align them (optional)
    array1 = np.roll(img_1.get_fdata(), 0, axis=0)
    array2 = np.roll(img_2.get_fdata(), 0, axis=0)

    # Identifying elements greater than 1000 in each array (data specific)
    mask1 = array1 > 1000
    mask2 = array2 > 1000

    # Create a result array, initially filled with zeros
    result = np.zeros_like(array1, dtype=float)

    # Case 1: Values from array1 where only array1 is greater than 1000
    result += np.where(mask1 & ~mask2, array1, 0)

    # Case 2: Values from array2 where only array2 is greater than 1000
    result += np.where(~mask1 & mask2, array2, 0)

    # Case 3: Highest value where both array1 and array2 are greater than 1000
    highest_values = np.maximum(array1, array2)
    result += np.where(mask1 & mask2, highest_values, 0)

    nib.save(nib.Nifti1Image(result, img_1.affine), args.out_file)


if __name__ == '__main__':
    main()
