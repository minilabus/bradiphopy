# -*- coding: utf-8 -*-
"""
Functions to facilitate operations with surfaces and their additional data.
"""

from functools import reduce

import numpy as np
from scipy.spatial import cKDTree
import vtk

from bradiphopy.bradipho_helper import BraDiPhoHelper3D


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


def apply_intersection(bdp_list):
    points_list = [bdp.get_polydata_vertices() for bdp in bdp_list]
    # Hash the points using the desired precision.
    indices = np.cumsum([0] + [len(p) for p in points_list[:-1]])
    hashes = [hash_points(s, i, precision=9) for
              s, i in zip(points_list, indices)]

    # Perform the operation on the hashes and get the output points.
    to_keep = reduce(intersection, hashes)
    indices = np.array(sorted(to_keep.values())).astype(np.uint32)
    bdp_out = bdp_list[0].subsample_polydata_vertices(indices)
    return bdp_out


def apply_union(bdp_list):
    append_filter = vtk.vtkAppendPolyData()
    for bdp_obj in bdp_list:
        append_filter.AddInputData(bdp_obj.get_polydata())
    append_filter.Update()
    bdp = BraDiPhoHelper3D(append_filter.GetOutput())
    bdp_out = apply_intersection([bdp])
    return bdp_out


def apply_difference(bdp_list):
    points_list = [bdp.get_polydata_vertices() for bdp in bdp_list]
    # Hash the points using the desired precision.
    indices = np.cumsum([0] + [len(p) for p in points_list[:-1]])
    hashes = [hash_points(s, i, precision=9) for
              s, i in zip(points_list, indices)]

    # Perform the operation on the hashes and get the output points.
    to_keep = reduce(intersection, hashes)
    indices = np.array(sorted(to_keep.values())).astype(np.uint32)
    ori_indices = np.arange(len(points_list[0]))
    indices = np.setdiff1d(ori_indices, indices)
    bdp_out = bdp_list[0].subsample_polydata_vertices(indices)

    return bdp_out
