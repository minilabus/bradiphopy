# -*- coding: utf-8 -*-
"""
This module provides a collection of operations for processing and manipulating
3D surface data (often represented as `BraDiPhoHelper3D` objects) and point
clouds. Functions include methods for transferring annotations, point set
registration (ICP), finding neighboring points, and set operations on point
clouds (intersection, union, difference).
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
    """
    Transfer annotations from source objects to a target object.

    Parameters
    ----------
    src_bdp_obj : list of BraDiPhoHelper3D
        List of source objects from which to transfer annotations.
    tgt_bdp_obj : BraDiPhoHelper3D
        Target object to which annotations will be transferred.
    distance : float, optional
        Maximum distance to consider a point in `tgt_bdp_obj` close to
        `src_bdp_obj` for annotation transfer. Defaults to 0.001.
    filenames : list of str, optional
        List of filenames corresponding to `src_bdp_obj`. Used with `annot_lut`
        to assign specific annotation values. Defaults to None.
    annot_lut : str or dict, optional
        Path to a JSON file or a dictionary mapping keywords (from `filenames`)
        to annotation values. If None, annotations are assigned based on the
        order of `src_bdp_obj`. Defaults to None.

    Returns
    -------
    BraDiPhoHelper3D
        The target object `tgt_bdp_obj` with updated 'annotation' scalar data.
    numpy.ndarray
        The array of new annotations applied to `tgt_bdp_obj`.
    """
    if annot_lut and isinstance(annot_lut, str):
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
                    logging.info('Found key {} for {}'.format(annot_key,
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

    Parameters
    ----------
    A : numpy.ndarray
        Source point cloud, shape (N, 3).
    B : numpy.ndarray
        Target point cloud, shape (M, 3).
    max_distance : float
        Maximum distance for closest point matching.
    iteration : int, optional
        The maximum number of ICP iterations. Defaults to 10.

    Returns
    -------
    tuple
        A tuple containing:
            - numpy.ndarray: The final transformation matrix (4x4).
            - numpy.ndarray: The transformed source point cloud (shape N, 3).
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
    """
    Find points in `tgt_bdp_obj` that are neighbors of `src_bdp_obj`.

    The function filters points in `tgt_bdp_obj` based on proximity to
    `src_bdp_obj` using bounding box, convex hull, and finally a KD-tree
    search after an initial ICP alignment.

    Parameters
    ----------
    src_bdp_obj : BraDiPhoHelper3D
        Source object.
    tgt_bdp_obj : BraDiPhoHelper3D
        Target object.
    max_dist : float, optional
        Maximum distance to consider points as neighbors. Defaults to 1.
    return_indices : bool, optional
        If True, also return the indices of the matched points in the
        original target object. Defaults to False.
    return_distances : bool, optional
        If True, also return the distances of the matched points.
        Defaults to False.

    Returns
    -------
    list
        A list containing the modified `tgt_bdp_obj` (subsampled to include
        only matching points).
        If `return_indices` is True, the list also contains `numpy.ndarray`
        of indices.
        If `return_distances` is True, the list also contains `numpy.ndarray`
        of distances of the matched points from `src_bdp_obj`.
    """
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
    """
    Produces a dict from points by using the points as keys and the
    indices of the points as values.

    Parameters
    ----------
    points : numpy.ndarray
        The array of points (Nx3) used to produce the dict.
    start_index : int, optional
        The starting index for the point values in the dictionary.
        Defaults to 0.
    precision : int, optional
        The number of decimals to keep when hashing the points.
        Allows a soft comparison of points. If None, no rounding is performed.
        Defaults to None.

    Returns
    -------
    dict
        A dict where the keys are byte representations of rounded points and
        the values are their original indices incremented by `start_index`.
    """

    keys = np.round(points, precision)
    keys.flags.writeable = False

    return {k.data.tobytes(): i for i, k in enumerate(keys, start_index)}


def intersection(left, right):
    """
    Computes the intersection of two dictionaries created by `hash_points`.

    The keys are byte representations of points, and values are their indices.

    Parameters
    ----------
    left : dict
        The first dictionary.
    right : dict
        The second dictionary.

    Returns
    -------
    dict
        A new dictionary containing only the key-value pairs present in both
        `left` and `right`.
    """
    return {k: v for k, v in left.items() if k in right}


def apply_intersection(bdp_list, return_indices=False):
    """
    Computes the intersection of multiple BraDiPhoHelper3D objects.

    Points are considered the same if their coordinates match up to a
    precision of 9 decimal places.

    Parameters
    ----------
    bdp_list : list of BraDiPhoHelper3D
        A list of `BraDiPhoHelper3D` objects.
    return_indices : bool, optional
        If True, also return the indices of the intersection points relative
        to the concatenated point list from the first object that contains
        all intersection points. Defaults to False.

    Returns
    -------
    BraDiPhoHelper3D
        A new `BraDiPhoHelper3D` object containing points present in all
        input objects.
    numpy.ndarray, optional
        If `return_indices` is True, an array of indices of the
        intersection points. These indices refer to the vertices in the
        first `BraDiPhoHelper3D` object in `bdp_list` if all intersection
        points are found within it.
    """
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
    """
    Computes the union of multiple BraDiPhoHelper3D objects.

    Duplicate points (matching up to 9 decimal places) are removed.

    Parameters
    ----------
    bdp_list : list of BraDiPhoHelper3D
        A list of `BraDiPhoHelper3D` objects.
    return_indices : bool, optional
        If True, also return the unique indices of the points in the union,
        relative to the concatenated list of all input points before
        uniqueness is enforced. Defaults to False.

    Returns
    -------
    BraDiPhoHelper3D
        A new `BraDiPhoHelper3D` object containing all unique points from the
        input objects.
    numpy.ndarray, optional
        If `return_indices` is True, an array of unique indices.
    """
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
    """
    Computes the difference between the first BraDiPhoHelper3D object and the
    union of the rest of the objects in the list.

    Points are considered the same if their coordinates match up to a
    precision of 9 decimal places.

    Parameters
    ----------
    bdp_list : list of BraDiPhoHelper3D
        A list of `BraDiPhoHelper3D` objects. The first element is the
        object from which the union of the subsequent objects will be
        subtracted.

    Returns
    -------
    BraDiPhoHelper3D
        A new `BraDiPhoHelper3D` object containing points from the first
        object that are not present in any of the subsequent objects.
    """
    bdp_union = apply_union(bdp_list[1:])
    # Find points in the first object that are also in the union of the rest
    _, indices_to_remove = apply_intersection([bdp_list[0], bdp_union],
                                              return_indices=True)

    # Get all indices from the first object
    ori_indices = np.arange(len(bdp_list[0]))
    # Determine indices to keep by finding those not in indices_to_remove
    indices = np.setdiff1d(ori_indices, indices_to_remove)
    bdp_out = bdp_list[0].subsample_polydata_vertices(indices)

    return bdp_out
    indices = np.setdiff1d(ori_indices, indices)
    bdp_out = bdp_list[0].subsample_polydata_vertices(indices)

    return bdp_out
