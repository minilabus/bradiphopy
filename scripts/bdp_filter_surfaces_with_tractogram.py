#!/usr/bin/env python3
#  -*- coding: utf-8 -*-

"""
If you want to compare multiple tractograms with a single surface,
you can run the script multiple times with different tractograms.

If you want to compare multiple surfaces with a single tractogram,
you can run the script multiple times with different surfaces."
"""

import argparse
from bradiphopy.io import load_polydata, save_polydata
from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.io.stateful_tractogram import (set_sft_logger_level,
                                         StatefulTractogram, Space)
from bradiphopy.bradipho_helper import BraDiPhoHelper3D
import numpy as np
from bradiphopy.segment import get_proximity_scores
from sklearn.cluster import KMeans
import os
import shutil

MODES = ['any', 'all', 'either_end', 'both_ends']
CRITERIA = ['include', 'exclude']


def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)

    p.add_argument('in_surfaces', nargs='+',
                   help='Path of the input surface files.')
    p.add_argument('in_tractograms', nargs='+',
                   help='Path of the input tractogram file (only support .trk).')

    p.add_argument('--max_distance', type=float, default=5,
                   help='Maximum distance to consider a streamline as '
                   'close to a surface. Default: 5.')
    p.add_argument('--endpoints_only', action='store_true',
                   help='Consider only the endpoints of the streamlines '
                   'to compute the proximity scores.')

    p.add_argument('--out_dir',
                   help='Path of the output directory for symlinks.')
    g = p.add_mutually_exclusive_group()
    g.add_argument('--print_top_N', type=int,
                   help='Print the top N surfaces above the mean in both.')
    g.add_argument('--print_best_cluster', action='store_true',
                   help='Print the best cluster of surfaces above the mean in both.')

    return p


def _get_proximity_scores_wrapper(sft, bdp_obj, distance=1, endpoints_only=False):
    surf_coverage, sft_coverage = get_proximity_scores(
        sft, bdp_obj, distance=distance,
        endpoints_only=endpoints_only)
    return surf_coverage, sft_coverage


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    all_files = args.in_surfaces + args.in_tractograms
    args.in_tractograms = [f for f in all_files if os.path.splitext(f)[
        1] == '.trk']
    args.in_surfaces = [f for f in all_files if os.path.splitext(f)[
        1] != '.trk']

    if len(args.in_surfaces) > 1 and len(args.in_tractograms) > 1:
        raise ValueError('Either pick multiple surface or multiple tractograms'
                         ', not both.')

    if os.path.isdir(args.out_dir):
        shutil.rmtree(args.out_dir)
    if args.out_dir and not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    score_result = {}
    for tractogram_name in args.in_tractograms:
        sft = load_tractogram(tractogram_name, 'same')

        for surf_name in args.in_surfaces:
            polydata = load_polydata(surf_name, to_lps=True)
            bdp_obj = BraDiPhoHelper3D(polydata)

            surf_coverage, sft_coverage = _get_proximity_scores_wrapper(
                sft, bdp_obj, distance=args.max_distance,
                endpoints_only=args.endpoints_only)
            if len(args.in_tractograms) > 1:
                score_result[tractogram_name] = (surf_coverage, sft_coverage)
            else:
                score_result[surf_name] = (surf_coverage, sft_coverage)

            if all([x[0] == 0 and x[1] == 0 for x in score_result.values()]):
                raise ValueError('No streamlines are close to any surface.')

    if args.print_top_N:
        mean_surf_coverage = np.mean([x[0] for x in score_result.values()])
        mean_sft_coverage = np.mean([x[1] for x in score_result.values()])
        top_both = [k for k, v in score_result.items()
                    if v[0] > mean_surf_coverage and v[1] > mean_sft_coverage]
        top_both = sorted(top_both, key=lambda x: score_result[x][0],
                          reverse=True)
        top_both = top_both[:args.print_top_N]
    elif args.print_best_cluster:
        list_of_scores = list(score_result.values())
        X = np.array(list_of_scores)

        kmeans = KMeans(n_clusters=2, random_state=0,
                        n_init='auto').fit(X)
        labels = kmeans.labels_
        # check if it is cluster 0 or 1 that has the highest mean
        mean_0 = np.mean([x[0] for i, x in enumerate(
            list_of_scores) if labels[i] == 0])
        mean_1 = np.mean([x[0] for i, x in enumerate(
            list_of_scores) if labels[i] == 1])
        labels_to_pick = 0 if mean_0 > mean_1 else 1
        top_both = [k for i, k in enumerate(
            score_result.keys()) if labels[i] == labels_to_pick]
    else:
        top_both = score_result.keys()

    print('Top files above mean in both:')
    for filename in top_both:
        if args.out_dir:
            in_filename = os.path.abspath(filename)
            out_filename = os.path.join(
                args.out_dir, os.path.basename(filename))
            os.symlink(in_filename, out_filename)

        # Printing results
        if len(args.in_tractograms) > 1:
            print('\tTractogram: {}'.format(filename))
        else:
            print('\tSurface: {}'.format(filename))
        scores = score_result[filename]
        print('\tSurface coverage: {}'.format(scores[0]))
        print('\tTractogram coverage: {}'.format(scores[1]))
        print()


if __name__ == "__main__":
    main()
