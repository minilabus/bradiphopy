#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to compute matching scores between tractography bundles and atlas labels
using endpoint distance maps.

This can be used to match bundles to stimulation targets in the brain for
example. We recommand spliting your endpoints into head and tail yourself,
an easy way to do this is to multiply your target mask by a Freesurfer
parcellation and use the resulting image as input.
"""

import argparse

from dipy.io.streamline import load_tractogram
import nibabel as nib
import numpy as np


from scilpy.tractograms.streamline_and_mask_operations import (
    get_head_tail_density_maps)
from scilpy.image.volume_operations import compute_distance_map


def _build_arg_parser():
    """
    Build and return the argument parser.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('in_endpoints',
                        help='Input atlas endpoints image.')
    parser.add_argument('in_bundles', nargs='+',
                        help='Input tractography bundle files.')
    return parser


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Load atlas endpoints image
    atlas_img = nib.load(args.in_endpoints)
    atlas_data = atlas_img.get_fdata()

    # Create binary mask of the atlas
    atlas_binary = np.zeros_like(atlas_data)
    atlas_binary[atlas_data > 0] = 1

    # Get unique labels in the atlas (excluding background)
    labels = np.unique(atlas_data)
    labels = labels[labels != 0].astype(int)

    endpoints = []
    for in_bundle in args.in_bundles:
        # Load tractogram
        sft = load_tractogram(in_bundle, 'same')

        # Get head and tail density maps & create endpoints mask
        head, tail = get_head_tail_density_maps(sft)
        endpoints_mask = np.zeros_like(atlas_data)
        endpoints_mask[head > 0] = 1
        endpoints_mask[tail > 0] = 1
        endpoints.append(endpoints_mask)

    # Initialize scores matrix (bundles x labels)
    scores = np.zeros((len(endpoints), len(labels)), dtype=float)

    # Compute scores between each endpoint mask and atlas labels
    for i, endpoint in enumerate(endpoints):
        distance_map = compute_distance_map(endpoint, atlas_binary,
                                            max_distance=10,
                                            symmetric=True)

        for j, label in enumerate(labels):
            curr_distance_map = distance_map.copy()
            curr_distance_map = np.where((endpoint == 0) & (atlas_data == 0),
                                         np.inf, curr_distance_map)
            curr_distance_map[atlas_data != label] = np.inf

            # Compute sum of exponential negative distances
            sum_dist = np.exp(-curr_distance_map)
            sum_dist = np.sum(sum_dist)
            scores[i, j] = sum_dist

    # Compute cost matrix by summing scores for each bundle
    scores[scores < 1e-3] -= np.mean(scores[scores > 1e-3])
    scores = scores.astype(int)
    cost_matrix = np.sum(scores, axis=1)
    indices = np.argsort(cost_matrix)

    np.set_printoptions(suppress=True)
    for ind in indices:
        print(f'Bundle: {args.in_bundles[ind]}, Cost: {cost_matrix[ind]}')
        print(f'Labels: {labels}, Scores: {scores[ind, :]}')
        print()


if __name__ == "__main__":
    main()
