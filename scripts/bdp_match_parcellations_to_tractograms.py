#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute the equivalent of the connectivity matrix for a set of surfaces and a
tractogram. The connectivity matrix is built by computing the distance of each
streamline to the surfaces and assigning it to the closest one. The output is a
set of tractograms, one for each pair of surfaces.

Support multiple tractograms to avoid concatenating them. Multiple surfaces are
required to build the connectivity matrix.
"""

import argparse
from itertools import combinations
import os
import shutil

from dipy.io.streamline import load_tractogram, save_tractogram
from nibabel.affines import apply_affine
import numpy as np
from scipy.spatial import KDTree

from bradiphopy.bradipho_helper import BraDiPhoHelper3D
from bradiphopy.io import load_polydata


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument("in_tractograms", nargs="+", help="A list of tractograms.")
    p.add_argument("in_surfaces", nargs="+", help="A list of surfaces.")
    p.add_argument(
        "out_dir", help="Output directory."
    )  # TODO optional, use .h5 by default
    # p.add_argument('--in_annotation',
    #                help='Input annotation file only works with one '
    #                     'surface.')
    p.add_argument(
        "--max_distance",
        type=float,
        default=5,
        help="Maximum distance to consider a streamline as part of "
        "a surface [%(default)s].",
    )
    p.add_argument(
        "--keep_original_filename",
        action="store_true",
        help="Keep the original filename of the surfaces.",
    )
    p.add_argument(
        "--save_empty", action="store_true", help="Save empty tractograms."
    )
    p.add_argument(
        "-f",
        dest="overwrite",
        action="store_true",
        help="Force overwriting of the output files.",
    )
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    in_files = args.in_tractograms + args.in_surfaces

    in_tractograms = [f for f in in_files if os.path.splitext(f)[1] == ".trk"]
    in_surfaces = [f for f in in_files if os.path.splitext(f)[1] != ".trk"]

    if os.path.isdir(args.out_dir) and not args.overwrite:
        raise ValueError(f"{args.out_dir} already exists. Use -f to overwrite.")
    elif os.path.isdir(args.out_dir) and args.overwrite:
        shutil.rmtree(args.out_dir)

    if not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    surfaces = []
    for surface in in_surfaces:
        polydata = load_polydata(surface)
        bdp_obj = BraDiPhoHelper3D(polydata)
        surfaces.append(bdp_obj)

    # Concatenate tractograms
    sft = load_tractogram(in_tractograms[0], "same")
    for in_tractogram in in_tractograms[1:]:
        sft += load_tractogram(in_tractogram, "same")

    # Get ready to compute distances from both ends of the streamlines
    distances = np.ones((len(surfaces), len(sft.streamlines), 2)) * 1e3
    heads = np.array([sft.streamlines[i][0] for i in range(len(sft.streamlines))])
    tail = np.array([sft.streamlines[i][-1] for i in range(len(sft.streamlines))])
    flip = np.diag([-1, -1, 1, 1])

    # Compute distances from both ends of the streamlines to all surfaces
    for i, surface in enumerate(surfaces):
        vertices = surface.get_polydata_vertices()
        vertices = apply_affine(flip, vertices)
        tree = KDTree(vertices)

        dist_1, _ = tree.query(heads, distance_upper_bound=args.max_distance)
        dist_2, _ = tree.query(tail, distance_upper_bound=args.max_distance)

        distances[i, :, 0] = dist_1
        distances[i, :, 1] = dist_2

    # We don't want to consider infinite distances
    distances[np.isinf(distances)] = 1e3

    # Find the closest surface for each streamline
    closest_1 = np.min(distances[:, :, 0], axis=0)
    closest_2 = np.min(distances[:, :, 1], axis=0)
    closest_1_pos = np.argmin(distances[:, :, 0], axis=0)
    closest_2_pos = np.argmin(distances[:, :, 1], axis=0)

    # If there was no closest surface, we set it to -1
    closest_1_pos[closest_1 == 1e3] = -1
    closest_2_pos[closest_2 == 1e3] = -1

    # Build the connectivity matrix
    comb_list = list(combinations(range(len(surfaces)), 2))
    comb_list += [(i, i) for i in range(len(surfaces))]

    for i, j in comb_list:
        # Since the matrix is symmetric, we only need to consider one half
        # Also, since endpoints are inversable, we check both combinations
        indices_pair = np.vstack([closest_1_pos, closest_2_pos]).T
        indices_1 = np.where((indices_pair[:, 0] == i) & (indices_pair[:, 1] == j))
        indices_2 = np.where((indices_pair[:, 0] == j) & (indices_pair[:, 1] == i))
        indices = np.unique(np.hstack([indices_1, indices_2]))

        if i > j:
            j, i = i, j

        if len(indices) == 0 and not args.save_empty:
            continue

        # Use labels to save the tractograms (easier for users)
        if args.keep_original_filename:
            basename_1 = os.path.splitext(os.path.basename(in_surfaces[i]))[0]
            basename_2 = os.path.splitext(os.path.basename(in_surfaces[j]))[0]
            save_tractogram(
                sft[indices],
                os.path.join(args.out_dir, f"{basename_1}_{basename_2}.trk"),
            )
        else:
            save_tractogram(sft[indices], os.path.join(args.out_dir, f"{i}_{j}.trk"))

        # TODO generated .h5
        # TODO Save bundle only if asked
        # TODO Support labels lists


if __name__ == "__main__":
    main()
