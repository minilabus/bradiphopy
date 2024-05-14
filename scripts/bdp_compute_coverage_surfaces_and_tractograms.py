#!/usr/bin/env python3
#  -*- coding: utf-8 -*-

"""
Compute the coverage of a tractogram to surfaces or the coverage of surfaces
to a tractogram.

The coverage is defined as the proportion of streamlines that are close to a
surface or the proportion of surface points that are close to a streamline.

The proximity is defined as the distance between a streamline and the closest
point on a surface. The distance is computed in mm.

--endpoints_only can be used to consider only the endpoints of the streamlines
to compute the proximity scores.
"""

import argparse
import json
import os
import shutil

from dipy.io.streamline import load_tractogram
import numpy as np
from sklearn.cluster import KMeans

from bradiphopy.bradipho_helper import BraDiPhoHelper3D
from bradiphopy.io import load_polydata
from bradiphopy.segment import get_proximity_scores


MODES = ['any', 'all', 'either_end', 'both_ends']
CRITERIA = ['include', 'exclude']


def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)

    p.add_argument('in_surfaces', nargs='+',
                   help='Path of the input surface files.')
    p.add_argument('in_tractograms', nargs='+',
                   help='Path of the input tractogram file (only support .trk).')
    p.add_argument('out_json',
                   help='Path of the output json file.')

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

    if args.out_dir and os.path.isdir(args.out_dir):
        shutil.rmtree(args.out_dir)
    if args.out_dir and not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    score_result = {}
    for tractogram_name in args.in_tractograms:
        sft = load_tractogram(tractogram_name, 'same')

        for surf_name in args.in_surfaces:
            polydata = load_polydata(surf_name, to_lps=True)
            bdp_obj = BraDiPhoHelper3D(polydata)

            surf_coverage, sft_coverage = get_proximity_scores(
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
        top_both = sorted(top_both, key=lambda x: top_both[x][0],
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
        top_both = sorted(score_result, key=lambda x: score_result[x][0],
                          reverse=True)

    print('Top files above mean in both:')
    for filename in top_both:
        if args.out_dir:
            in_filename = os.path.abspath(filename)
            out_filename = os.path.join(
                args.out_dir, os.path.basename(filename))
            os.symlink(in_filename, out_filename)

        # Printing results
        scores = score_result[filename]
        if sum(scores) < 1e-6:
            continue
        if len(args.in_tractograms) > 1:
            print('\tTractogram: {}'.format(filename))
        else:
            print('\tSurface: {}'.format(filename))
        print('\tSurface coverage: {}'.format(scores[0]))
        print('\tTractogram coverage: {}'.format(scores[1]))
        print()

    with open(args.out_json, 'w') as f:
        json.dump(score_result, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
