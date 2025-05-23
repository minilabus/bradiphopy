#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to compute matching scores between tractography bundles and labels
using a distances between all endpoints and all labels.

This can be used to match bundles to stimulation targets in the brain for
example. We recommand spliting your endpoints into head and tail yourself,
an easy way to do this is to multiply your target mask by a Lobe
parcellation and use the resulting image as input.

The script will output a pseudocore for each bundle, showing the labels
that are the most covered by the closest endpoints.
"""

import argparse

from dipy.io.streamline import load_tractogram
import nibabel as nib
import numpy as np


from scilpy.tractograms.streamline_and_mask_operations import (
    get_head_tail_density_maps)
from scilpy.image.volume_operations import compute_distance_map


def _build_arg_parser():
    """Builds and returns an argparse.ArgumentParser for this script.

    The parser is configured with arguments for:
    - Input atlas/labels image (NIFTI format).
    - One or more input tractography bundle files.
    - Maximum distance to consider for matching.
    - Optional number of top matches to display.
    The script's module-level docstring is used as the description.

    Returns:
        argparse.ArgumentParser: The configured argument parser.
    """
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_labels',
                   help='Input atlas endpoints image.')
    p.add_argument('in_bundles', nargs='+',
                   help='Input tractography bundle files.')

    p.add_argument('--max_distance', type=float, default=5,
                   help='Maximum distance to consider a streamline as part of '
                        'a surface [%(default)s].')
    p.add_argument('--show_top', type=int, const=5, nargs='?',
                   help='Number of top matches to show [%(default)s].')
    return p


def main():
    """Main function to match tractography bundles to atlas labels.

    Parses command-line arguments. Loads the input atlas image and extracts
    unique label values.
    For each input bundle:
        - Loads the tractogram.
        - Generates head and tail density maps using Scilpy's
          `get_head_tail_density_maps` to create a binary mask of the
          bundle's endpoints.
    Initializes a matrix to store matching scores (bundles x labels).
    Iterates through each bundle's endpoint mask and each label in the atlas:
        - Computes a distance map between the current bundle's endpoints
          and all labels in the atlas using Scilpy's `compute_distance_map`.
        - For the specific current label, isolates relevant distances from
          the map (distances between current bundle's endpoints and current
          label's voxels).
        - Calculates coverage proportions: `covered_endpoints` (how much of
          the bundle's endpoints are near the label) and `covered_atlas`
          (how much of the label is near the bundle's endpoints).
        - A `non_overlap_factor` (minimum of the two coverages) is computed.
        - A matching score is calculated based on the sum of exponentiated
          negative distances, normalized by the squared `non_overlap_factor`.
          This score reflects both proximity and extent of overlap.
    The raw scores are then transformed (logarithm and square) and summed
    per bundle to get a 'cost'. Bundles are ranked by this cost.
    Finally, it prints the matching scores for the top N bundles (or all,
    if `--show_top` is not specified), showing individual scores against
    each label and the total cost for the bundle.
    """
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Load atlas endpoints image
    labels_img = nib.load(args.in_labels)
    labels_data = labels_img.get_fdata()

    # Create binary mask of the atlas
    labels_binary = np.zeros_like(labels_data)
    labels_binary[labels_data > 0] = 1

    # Get unique labels in the atlas (excluding background)
    labels = np.unique(labels_data)
    labels = labels[labels != 0].astype(int)

    endpoints = []
    for in_bundle in args.in_bundles:
        # Load tractogram
        sft = load_tractogram(in_bundle, 'same')

        # Get head and tail density maps & create endpoints mask
        head, tail = get_head_tail_density_maps(sft)
        endpoints_mask = np.zeros_like(labels_data)
        endpoints_mask[head > 0] = 1
        endpoints_mask[tail > 0] = 1
        endpoints.append(endpoints_mask)

    # Initialize scores matrix (bundles x labels)
    scores = np.zeros((len(endpoints), len(labels)), dtype=float)

    # Compute scores between each endpoint mask and atlas labels
    if args.max_distance is None:
        args.max_distance = np.inf
    for i, endpoint in enumerate(endpoints):
        distance_map = compute_distance_map(endpoint, labels_binary,
                                            max_distance=args.max_distance,
                                            symmetric=True)

        for j, label in enumerate(labels):
            curr_distance_map = distance_map.copy()
            curr_distance_map = np.where((endpoint == 0) & (labels_data == 0),
                                         np.inf, curr_distance_map)
            curr_distance_map = np.where((labels_data != label) & (labels_data != 0),
                                         np.inf, curr_distance_map)

            # curr_distance_map[] = np.inf
            curr_bin_atlas = np.zeros_like(labels_data)
            curr_bin_atlas[labels_data == label] = 1

            covered_endpoints = np.sum(
                endpoint[~np.isinf(curr_distance_map)]) / np.sum(endpoint)
            covered_atlas = np.sum(curr_bin_atlas[~np.isinf(
                curr_distance_map)]) / np.sum(curr_bin_atlas)

            non_overlap_factor = min(covered_atlas, covered_endpoints)

            if non_overlap_factor == 0:
                scores[i, j] = 0
            else:
                sum_dist = np.exp(-1 * curr_distance_map /
                                  (non_overlap_factor ** 2))
                sum_dist = np.sum(sum_dist)
                scores[i, j] = sum_dist

    # Compute cost matrix by summing scores for each bundle
    scores[scores > 1e-3] = np.log(scores[scores > 1e-3]) ** 2
    scores = scores.astype(int)
    cost_matrix = np.sum(scores, axis=1)
    indices = np.argsort(cost_matrix)

    if args.show_top:
        indices = indices[-args.show_top:]

    np.set_printoptions(suppress=True)
    for ind in indices:
        print(f'Bundle: {args.in_bundles[ind]}, Cost: {cost_matrix[ind]}')
        print(f'Labels: {labels}, Scores: {np.round(scores[ind, :], 3)}')
        print()


if __name__ == "__main__":
    main()
