#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Combine multiple labels into a single file using a json file to map the labels.
Useful to change nomenclature between integers to names.

Utility script for the generate_all_parcellation_and_point_cloud.sh recipe.
(Not in this repository)
"""

import argparse
import json
import os

import nibabel as nib

from scilpy.image.labels import combine_labels


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument("in_folder", help="Input folder containing the nifti files.")
    p.add_argument(
        "in_json", help="Input json file containing the labels (mapping)."
    )
    p.add_argument("out_file", help="Output file (nifti).")
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if os.path.isfile(args.out_file) and not args.overwrite:
        raise IOError(
            "{} already exists, use -f to overwrite.".format(args.out_file)
        )

    with open(args.in_json) as f:
        lut = json.load(f)

    data_as_list = []
    values_as_list = []
    for key, value in lut.items():
        # Check if the file exists
        filepath = os.path.join(args.in_folder, key) + ".nii.gz"
        if not os.path.exists(filepath):
            print(f"File {filepath} does not exist.")
            continue

        # Generate input for scilpy function combine_labels
        img = nib.load(filepath)
        data = img.get_fdata()
        data_as_list.append(data)
        values_as_list.append(value)

    out_choice = ("out_labels_ids", values_as_list)
    indices_per_input_volume = [[1] for _ in range(len(data_as_list))]
    resulting_labels = combine_labels(
        data_as_list, indices_per_input_volume, out_choice
    )
    # Save the resulting labels
    nib.save(nib.Nifti1Image(resulting_labels, img.affine), args.out_file)


if __name__ == "__main__":
    main()
