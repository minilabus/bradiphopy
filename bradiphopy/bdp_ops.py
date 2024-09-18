# -*- coding: utf-8 -*-
"""
Functions to facilitate operations with surfaces and their additional data.
"""

from functools import reduce
import json
import logging
import os
import random

import numpy as np
from scipy.linalg import orthogonal_procrustes
from scipy.spatial import cKDTree, Delaunay, ConvexHull
import vtk

from bradiphopy.bradipho_helper import BraDiPhoHelper3D


def transfer_annots(src_bdp_obj, tgt_bdp_obj, distance=0.001,
                    filenames=None, annot_lut=None):

    if annot_lut:
        with open(annot_lut, 'r') as f:
            annot_lut = json.load(f)

    totals_size = np.cumsum([len(src) for src in src_bdp_obj])
    totals_size = np.insert(totals_size, 0, 0)
    merged_annots = np.zeros((totals_size[-1],), dtype=np.uint8)
    for i in range(len(totals_size)-1):
        if filenames and annot_lut:
            curr_name = os.path.basename(
                os.path.splitext(filenames[i])[0]).lower()
            for annot_key, annot_value in annot_lut.items():
                index_match = curr_name.find('_'+annot_key.lower())

                if index_match >= 0 and \
                        curr_name[index_match:] == '_'+annot_key.lower():
                    val = annot_value
                    print('Found key {} for {}'.format(annot_key,
                                                       filenames[i]))
                    break
            else:
                val = 1+i
        merged_annots[totals_size[i]:totals_size[i+1]] = val

    merged_vertices = np.vstack(
        [src.get_polydata_vertices() for src in src_bdp_obj])
    merged_ckd_tree = cKDTree(merged_vertices)

    # Get the surface closest point in the point cloud
    new_annots = np.zeros((len(tgt_bdp_obj),), dtype=np.uint8)
    distances, indices = merged_ckd_tree.query(tgt_bdp_obj.get_polydata_vertices(),
                                               k=1, distance_upper_bound=distance)
    indices[distances == np.inf] = -1
    for i, ind in enumerate(indices):
        new_annots[i] = 0 if ind == -1 else merged_annots[ind]

    tgt_bdp_obj.set_scalar(new_annots, name='annotation')
    return tgt_bdp_obj, new_annots


def run_icp(A, B, max_distance, iteration=10):
    """
    Run the Iterative Closest Point (ICP) algorithm.

    Parameters:
    A (numpy.ndarray): Source point cloud (Nx3).
    B (numpy.ndarray): Target point cloud (Mx3).
    max_distance (float): Maximum distance for closest point matching.

    Returns:
    tuple: Tuple containing:
        - numpy.ndarray: The final transformation matrix (4x4).
        - numpy.ndarray: The transformed source point cloud.
    """
    original_A = A.copy()
    if len(A) > 10000:
        A = A[np.random.choice(A.shape[0], 10000, replace=False), :]
    if len(B) > 10000:
        B = B[np.random.choice(B.shape[0], 10000, replace=False), :]
    assert A.shape[1] == 3 and B.shape[
        1] == 3, "Point clouds must have shape (N,3) or (M,3)"

    # Initial guess for the transformation
    R = np.eye(3)
    t = np.zeros(3)

    for i in range(iteration):  # Iteration limit
        # Apply current transformation
        A_transformed = np.dot(A, R.T) + t

        # Build a KD-Tree for efficient nearest neighbor search
        tree = cKDTree(B)
        distances, indices = tree.query(A_transformed,
                                        distance_upper_bound=max_distance)

        # Filter pairs with distances within the threshold
        valid_idx = np.where(distances < max_distance)[0]
        A_matched = A_transformed[valid_idx]
        B_matched = B[indices[valid_idx]]

        if len(A_matched) < 3 or len(B_matched) < 3:
            break  # Not enough points for a stable alignment

        # Compute the optimal rotation and translation (using Procrustes analysis)
        R, scale = orthogonal_procrustes(A_matched, B_matched)
        t = B_matched.mean(axis=0) - np.dot(A_matched.mean(axis=0), R)

    # Construct the final transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    return T, np.dot(original_A, R.T) + t


def match_neighbors(src_bdp_obj, tgt_bdp_obj, max_dist=1,
                    return_indices=False, return_distances=False):
    src_vectices = src_bdp_obj.get_polydata_vertices()
    tgt_vertices = tgt_bdp_obj.get_polydata_vertices()

    initial_size = len(tgt_vertices)
    logging.warning(
        'Number of vertices in source: {}'.format(len(src_vectices)))
    logging.warning(
        'Number of vertices in target: {}'.format(len(tgt_vertices)))

    src_bbox = np.array(src_bdp_obj.get_bound()).reshape(3, 2).T
    logging.warning('Source bounding box X: {} / Y: {} / Z: {}'.format(
        np.round(src_bbox[:, 0], 4),
        np.round(src_bbox[:, 1], 4),
        np.round(src_bbox[:, 2], 4)))

    # Extend the bbox by the max_distance per axis to make sure
    # we get all the vertices
    src_bbox[0] -= max_dist
    src_bbox[1] += max_dist
    min_condition = np.min(tgt_vertices - src_bbox[0], axis=1) > 0
    max_condition = np.max(tgt_vertices - src_bbox[1], axis=1) < 0
    bbox_in_indices = np.where(np.logical_and(min_condition,
                                              max_condition))[0]

    # Select the vertices in the bbox
    tgt_bdp_obj = tgt_bdp_obj.subsample_polydata(bbox_in_indices)
    tgt_vertices = tgt_vertices[bbox_in_indices]
    logging.warning('Number of vertices of target within source '
                    'bbox: {}'.format(len(bbox_in_indices)))

    # This makes the convex hull bigger and more stable for computation
    barycenter = np.mean(src_vectices, axis=0)
    tmp_convex_hull = src_vectices[ConvexHull(src_vectices).vertices]
    barycenter = np.mean(tmp_convex_hull, axis=0)
    big_convex_hull = tmp_convex_hull + (barycenter - tmp_convex_hull) * 0.1
    small_convex_hull = tmp_convex_hull - (barycenter - tmp_convex_hull) * 0.1
    new_convex_hull = np.vstack([big_convex_hull, small_convex_hull])

    convex_hull = new_convex_hull[ConvexHull(new_convex_hull).vertices]
    convex_hull = Delaunay(convex_hull)
    convex_hull_in_indices = [i for i in range(len(tgt_vertices))
                              if convex_hull.find_simplex(tgt_vertices[i]) >= 0]
    # Select the vertices in the convex hull
    tgt_bdp_obj = tgt_bdp_obj.subsample_polydata(
        convex_hull_in_indices)
    tgt_vertices = tgt_vertices[convex_hull_in_indices]

    # To help matching, we will run a small ICP first
    _, src_vectices = run_icp(src_vectices, tgt_vertices, max_dist)
    logging.warning('Number of vertices of target within convex hull of '
                    'source: {}'.format(len(convex_hull_in_indices)))

    src_ckd_tree = cKDTree(src_vectices)
    distances, _ = src_ckd_tree.query(tgt_vertices,
                                      k=1, distance_upper_bound=max_dist)
    # Select the vertices within the max distance
    close_indices = np.argwhere(distances < max_dist).flatten()
    tgt_bdp_obj = tgt_bdp_obj.subsample_polydata(close_indices)
    real_indices = bbox_in_indices[convex_hull_in_indices][close_indices]
    logging.warning('Number of vertices of target within max distance of '
                    'source: {}'.format(len(close_indices)))

    return_data = [tgt_bdp_obj]

    if return_indices:
        return_data.append(real_indices)
    if return_distances:
        real_distances = np.ones((initial_size)) * np.inf
        real_distances[real_indices] = distances[close_indices]
        return_data.append(distances[close_indices])

    return return_data


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
