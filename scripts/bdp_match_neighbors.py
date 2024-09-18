#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Match neighbors from one point_cloud to another.
Works better the closer the two files are in space.
"""

import argparse
import os
import shutil

import numpy as np
from scipy.spatial import cKDTree

from bradiphopy.io import load_polydata, save_polydata
from bradiphopy.bdp_ops import match_neighbors
from bradiphopy.bradipho_helper import BraDiPhoHelper3D


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('src_files', nargs='+',
                   help='Annotated source filename (must be supported by VTK).')
    p.add_argument('tgt_file',
                   help='Unannotated target filename (must be supported by VTK).')

    p.add_argument('--out_suffix', default='_matched',
                   help='Output suffix (must be supported by VTK) [%(default)s].')
    p.add_argument('--out_dir',
                   help='Output suffix (must be supported by VTK) [./].')

    p.add_argument('--distance', type=float, default=0.005,
                   help='Maximum distance for transfer (mm) [%(default)s]')
    p.add_argument('--fix_unannotated', action='store_true',
                   help='Attempt to fix unannotated points with closest '
                        'neighbors. \nOnly work when every vertice is expected '
                        'to have a label.')
    p.add_argument('--ascii', action='store_true',
                   help='Save the file with data as ASCII '
                        '(instead of binary).')
    p.add_argument('-f', dest='overwrite', action='store_true',
                   help='Force overwriting of the output files.')
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.out_dir is not None and os.path.isdir(args.out_dir):
        if not args.overwrite:
            raise IOError(
                '{} already exists, use -f to overwrite.'.format(args.out_dir))
        else:
            shutil.rmtree(args.out_dir)
            os.makedirs(args.out_dir)
    elif args.out_dir is not None and not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)
    else:
        args.out_dir = './'

    for i, src_file in enumerate(args.src_files):
        basename = os.path.basename(src_file)
        out_filename = os.path.join(args.out_dir,
                                    os.path.splitext(basename)[0] + args.out_suffix + '.ply')
        if os.path.isfile(out_filename) and not args.overwrite:
            raise IOError(
                '{} already exists, use -f to overwrite.'.format(out_filename))

    tgt_polydata = load_polydata(args.tgt_file)
    tgt_bdp_obj = BraDiPhoHelper3D(tgt_polydata)
    annotations = np.zeros((len(tgt_bdp_obj),), dtype=np.uint8)
    distances = np.ones((len(tgt_bdp_obj),), dtype=np.float32) * np.inf

    for i, src_file in enumerate(args.src_files):
        src_polydata = load_polydata(src_file)
        src_bdp_obj = BraDiPhoHelper3D(src_polydata)

        _, curr_indices, curr_distances = match_neighbors(src_bdp_obj,
                                                          tgt_bdp_obj,
                                                          max_dist=args.distance,
                                                          return_indices=True,
                                                          return_distances=True)
        print()
        # Where distances are smaller, replace the annotation
        # and the distance
        replace_indices = curr_distances < distances[curr_indices]
        real_indices = curr_indices[replace_indices]

        annotations[real_indices] = i+1
        distances[real_indices] = curr_distances[replace_indices]

    vertices = tgt_bdp_obj.get_polydata_vertices()
    annotated = np.where(annotations != 0)[0]
    # Create a KDTree with the annotated vertices
    tree = cKDTree(vertices[annotated])

    # Filter out the vertices with annotation 0
    valid_vertices = vertices[annotations != 0]

    # Query the tree for the nearest neighbors of the valid vertices
    distances, indices = tree.query(valid_vertices, k=10,
                                    distance_upper_bound=args.distance * 5)

    # Filter out indices beyond the distance threshold
    valid_indices = indices[distances != np.inf]

    # Apply bincount and argmax to find the most common annotation for each vertex
    # Reshape indices to match the shape of annotations for valid vertices
    reshaped_indices = annotations[annotated][valid_indices].reshape(
        valid_vertices.shape[0], -1)

    new_annotations = np.array([np.argmax(np.bincount(row))
                                for row in reshaped_indices])
    final_annotations = np.zeros_like(annotations, dtype=np.uint8)
    final_annotations[annotations != 0] = new_annotations

    annotations = final_annotations

    if args.fix_unannotated:
        unannotated = np.where(annotations == 0)[0]
        annotated = np.where(annotations != 0)[0]

        tree = cKDTree(vertices[annotated])
        _, indices = tree.query(vertices[unannotated], k=1)
        annotations[unannotated] = annotations[annotated[indices]]

    for i, src_file in enumerate(args.src_files):
        final_indices = np.where(annotations == i+1)[0]
        basename = os.path.basename(src_file)
        out_filename = os.path.join(args.out_dir,
                                    os.path.splitext(basename)[0] + args.out_suffix + '.ply')

        if len(final_indices):
            tmp_bdp_obj = tgt_bdp_obj.subsample_polydata(final_indices)
            save_polydata(tmp_bdp_obj.get_polydata(),
                          out_filename, ascii=args.ascii)
        else:
            print('No points matched for {}'.format(src_file))


if __name__ == '__main__':
    main()
