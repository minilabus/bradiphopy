# -*- coding: utf-8 -*-
"""
Functions to facilitate operations with surfaces and their additional data.
"""

from functools import reduce
import json
import os

import numpy as np
from scipy.spatial import cKDTree
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

        if annot_lut:
            for key, value in annot_lut.items():
                if key.lower() in curr_key.lower():
                    curr_key = value
                    print('Found key {} for {}'.format(key, filenames[i]))
                    break

        _, indices[curr_key] = ckd_tree.query(src.get_polydata_vertices(),
                                              k=1, distance_upper_bound=distance)

    new_annots = np.zeros((len(tgt_bdp_obj),), dtype=np.uint8)
    for i, tuple_key_val in enumerate(indices.items()):
        key, idx = tuple_key_val
        if isinstance(key, int):
            new_annots[idx] = key
        else:
            new_annots[idx] = i+1

    tgt_bdp_obj.set_scalar(new_annots, name='annotation')
    return tgt_bdp_obj, new_annots


def match_neighbors(src_bdp_obj, tgt_bdp_obj, distance=1):
    ckd_tree = cKDTree(tgt_bdp_obj.get_polydata_vertices())
    _, indices = ckd_tree.query(src_bdp_obj.get_polydata_vertices(),
                                k=10, distance_upper_bound=distance)

    return tgt_bdp_obj.subsample_polydata_vertices(np.unique(indices))


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
