# -*- coding: utf-8 -*-
"""
Functions to facilitate operations with surfaces and their additional data.
"""

from functools import reduce
import json
import logging
import os

import numpy as np
from scipy.spatial import cKDTree, Delaunay, ConvexHull
import vtk

from bradiphopy.bradipho_helper import BraDiPhoHelper3D


def transfer_annots(src_bdp_obj, tgt_bdp_obj, distance=1, filenames=None,
                    annot_lut=None):
    ckd_tree = cKDTree(tgt_bdp_obj.get_polydata_vertices())

    if annot_lut:
        with open(annot_lut) as f:
            annot_lut = json.load(f)

    indices = {}
    for i, src in enumerate(src_bdp_obj):
        curr_key = os.path.basename(
            os.path.splitext(filenames[i])[0]) if filenames else i

        # Check if the key is in the LUT
        if annot_lut:
            for key, value in annot_lut.items():
                if key.lower() in curr_key.lower():
                    curr_key = value
                    print('Found key {} for {}'.format(key, filenames[i]))
                    break

        # Get the surface closest point in the point cloud
        _, indices[curr_key] = ckd_tree.query(src.get_polydata_vertices(),
                                              k=1, distance_upper_bound=distance)

    # Assign the indices to the new annotation (LUT or not)
    new_annots = np.zeros((len(tgt_bdp_obj),), dtype=np.uint8)
    for i, tuple_key_val in enumerate(indices.items()):
        key, idx = tuple_key_val
        if isinstance(key, int):
            new_annots[idx] = key
        else:
            new_annots[idx] = i+1

    tgt_bdp_obj.set_scalar(new_annots, name='annotation')
    return tgt_bdp_obj, new_annots


def match_neighbors(src_bdp_obj, tgt_bdp_obj, max_dist=1):
    src_vectices = src_bdp_obj.get_polydata_vertices()
    tgt_vertices = tgt_bdp_obj.get_polydata_vertices()
    logging.warning(
        'Number of vertices in source: {}'.format(len(src_vectices)))
    logging.warning(
        'Number of vertices in target: {}'.format(len(tgt_vertices)))

    src_bbox = np.array(src_bdp_obj.get_bound()).reshape(3, 2).T
    logging.warning('Source bounding box X: {} / Y: {} / Z: {}'.format(
        np.round(src_bbox[:, 0], 4),
        np.round(src_bbox[:, 1], 4),
        np.round(src_bbox[:, 2], 4)))
    min_condition = np.min(tgt_vertices-src_bbox[0], axis=1) > 0
    max_condition = np.any(tgt_vertices-src_bbox[1], axis=1) < 0
    bbox_in_indices = np.where(np.logical_or(min_condition,
                                             max_condition))[0]
    # Select the vertices in the bbox
    tgt_bdp_obj = tgt_bdp_obj.subsample_polydata_vertices(bbox_in_indices)
    tgt_vertices = tgt_vertices[bbox_in_indices]
    logging.warning('Number of vertices of target within source '
                    'bbox: {}'.format(len(bbox_in_indices)))

    convex_hull = src_vectices[ConvexHull(src_vectices).vertices]*1.1
    convex_hull = Delaunay(convex_hull)
    convex_hull_in_indices = [i for i in range(len(tgt_vertices))
                              if convex_hull.find_simplex(tgt_vertices[i]) >= 0]
    # Select the vertices in the convex hull
    tgt_bdp_obj = tgt_bdp_obj.subsample_polydata_vertices(
        convex_hull_in_indices)
    tgt_vertices = tgt_vertices[convex_hull_in_indices]
    logging.warning('Number of vertices of target within convex hull of '
                    'source: {}'.format(len(convex_hull_in_indices)))

    src_ckd_tree = cKDTree(src_vectices)
    distances, _ = src_ckd_tree.query(tgt_vertices,
                                      k=1, distance_upper_bound=max_dist)
    # Select the vertices within the max distance
    close_indices = np.argwhere(distances < max_dist).flatten()
    tgt_bdp_obj = tgt_bdp_obj.subsample_polydata_vertices(close_indices)
    logging.warning('Number of vertices of target within max distance of '
                    'source: {}'.format(len(close_indices)))

    return tgt_bdp_obj


def hash_points(points, start_index=0, precision=None):
    """Produces a dict from points

    Produces a dict from points by using the points as keys and the
    indices of the points as values.

    Parameters
    ----------
    points: list of ndarray
        The list of points used to produce the dict.
    start_index: int, optional
        The index of the first streamline. 0 by default.
    precision: int, optional
        The number of decimals to keep when hashing the points of the
        points. Allows a soft comparison of points. If None, no
        rounding is performed.

    Returns
    -------
    A dict where the keys are streamline points and the values are indices
    starting at start_index.

    """

    keys = np.round(points, precision)
    keys.flags.writeable = False

    return {k.data.tobytes(): i for i, k in enumerate(keys, start_index)}


def intersection(left, right):
    """Intersection of two points dict (see hash_points)"""
    return {k: v for k, v in left.items() if k in right}


def apply_intersection(bdp_list, return_indices=False):
    points_list = [bdp.get_polydata_vertices() for bdp in bdp_list]
    # Hash the points using the desired precision.
    indices = np.cumsum([0] + [len(p) for p in points_list[:-1]])
    hashes = [hash_points(s, i, precision=9) for
              s, i in zip(points_list, indices)]

    # Perform the operation on the hashes and get the output points.
    to_keep = reduce(intersection, hashes)
    indices = np.array(sorted(to_keep.values())).astype(np.uint32)
    bdp_out = bdp_list[0].subsample_polydata_vertices(indices)

    if return_indices:
        return bdp_out, indices
    else:
        return bdp_out


def apply_union(bdp_list, return_indices=False):
    append_filter = vtk.vtkAppendPolyData()
    for bdp_obj in bdp_list:
        append_filter.AddInputData(bdp_obj.get_polydata())
    append_filter.Update()
    bdp = BraDiPhoHelper3D(append_filter.GetOutput())
    bdp_out, indices = apply_intersection([bdp], return_indices=True)

    if return_indices:
        return bdp_out, indices
    else:
        return bdp_out


def apply_difference(bdp_list):
    bdp_union = apply_union(bdp_list[1:])
    bdp_out, indices = apply_intersection([bdp_list[0], bdp_union],
                                          return_indices=True)

    ori_indices = np.arange(len(bdp_list[0]))
    indices = np.setdiff1d(ori_indices, indices)
    bdp_out = bdp_list[0].subsample_polydata_vertices(indices)

    return bdp_out
