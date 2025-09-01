# -*- coding: utf-8 -*-
"""
Provides functions for segmenting and filtering neurological tractography data
(streamlines) based on their spatial relationship with 3D surfaces.
"""

import numpy as np

from scipy.spatial import KDTree
from scilpy.tractograms.streamline_operations import resample_streamlines_num_points


def filter_from_surface(sft, bdp_obj, mode, criteria, distance, matched_pts=None):
    """
    Filters streamlines based on their proximity to a given surface object.

    Parameters
    ----------
    sft : scilpy.io.streamlines.StatefulTractogram
        The input StatefulTractogram object containing streamlines.
    bdp_obj : bradiphopy.bradipho_helper.BraDiPhoHelper3D
        The surface object used for filtering.
    mode : str
        Specifies how streamlines are selected based on point proximity.
        - 'any': Select streamline if any of its points are within `distance`
                 to the surface.
        - 'all': Select streamline if all of its points are within `distance`
                 to the surface.
        - 'either_end': Select streamline if at least one of its endpoints is
                        within `distance` to the surface.
        - 'both_ends': Select streamline if both of its endpoints are within
                       `distance` to the surface.
    criteria : str
        Determines how to use the `mode` condition for filtering.
        - 'include': Keep streamlines that meet the `mode` condition.
        - 'exclude': Remove streamlines that meet the `mode` condition.
    distance : float
        Maximum distance (exclusive) for a point on a streamline to be
        considered near the surface.
    matched_pts : numpy.ndarray, optional
        A boolean array indicating points from `sft.streamlines._data` that
        have already been matched in a previous step. These points will be
        ignored (their distance effectively set to infinity). Defaults to None.

    Returns
    -------
    tuple
        - numpy.ndarray: Indices of the streamlines in the input `sft` that
                         meet the filtering conditions.
        - numpy.ndarray or None: Updated boolean array indicating points from
                                 `sft.streamlines._data` that are close to the
                                 surface. Returns `None` if input `matched_pts`
                                 was `None`. Otherwise, it's a boolean array
                                 where True means a point is within `distance`.
    """
    sft_pts = sft.streamlines._data
    surf_pts = bdp_obj.get_polydata_vertices()

    tree = KDTree(surf_pts)
    dist, _ = tree.query(sft_pts, k=1, distance_upper_bound=distance)

    if matched_pts is not None:
        dist[matched_pts] = np.inf

    indices = []
    tmp_len = [len(s) for s in sft.streamlines]
    offsets = np.insert(np.cumsum(tmp_len), 0, 0)

    for i in range(len(offsets) - 1):
        curr_distance = dist[offsets[i] : offsets[i + 1]]
        if mode == "any" and np.any(curr_distance != np.inf):
            indices.append(i)
        elif mode == "all" and np.all(curr_distance != np.inf):
            indices.append(i)
        elif mode == "either_end" and (
            curr_distance[0] != np.inf or curr_distance[-1] != np.inf
        ):
            indices.append(i)
        elif mode == "both_ends" and (
            curr_distance[0] != np.inf and curr_distance[-1] != np.inf
        ):
            indices.append(i)

    if criteria == "exclude":
        indices = np.setdiff1d(np.arange(len(sft)), indices)
    # If criteria is 'include' (or anything other than 'exclude'),
    # 'indices' already holds the streamlines matching the mode.

    # Update matched_pts based on current pass if it was provided
    current_pass_matched_pts = dist != np.inf
    if matched_pts is not None:
        # If initial matched_pts was provided, this means we are in an iterative process.
        # The returned matched_pts should reflect all points found so far *plus* current ones.
        # However, the function's current logic for input `matched_pts` is to ignore them
        # for distance calculation (dist[matched_pts] = np.inf).
        return indices, current_pass_matched_pts
    else:
        return indices, None


def get_proximity_scores(sft, bdp_obj, distance=1, endpoints_only=False):
    """
    Calculates proximity scores between streamlines and a surface object.

    The scores indicate the proportion of surface points close to streamlines
    and the proportion of streamline points (or endpoints) close to the surface.

    Parameters
    ----------
    sft : scilpy.io.streamlines.StatefulTractogram
        The input StatefulTractogram object.
    bdp_obj : bradiphopy.bradipho_helper.BraDiPhoHelper3D
        The surface object.
    distance : float, optional
        Radius (inclusive) for searching neighboring points between
        streamlines and surface. Defaults to 1.
    endpoints_only : bool, optional
        If True, resamples streamlines to only their two endpoints before
        calculating proximity. Defaults to False.

    Returns
    -------
    tuple
        - float: Proportion of surface points with at least one streamline point within `distance`.
        - float: Proportion of streamline points (or endpoints) with at least one surface point within `distance`.
    """
    if endpoints_only:
        sft = resample_streamlines_num_points(sft, 2)
    sft_pts = sft.streamlines._data
    surf_pts = bdp_obj.get_polydata_vertices()

    tree = KDTree(surf_pts)
    sft_ids = tree.query_ball_point(sft_pts, r=distance)

    tree = KDTree(sft_pts)
    surf_ids = tree.query_ball_point(surf_pts, r=distance)

    surf_coverage = np.array([1 if len(d) > 0 else 0 for d in surf_ids])
    sft_coverage = np.array([1 if len(d) > 0 else 0 for d in sft_ids])

    return np.sum(surf_coverage) / len(surf_pts), np.sum(sft_coverage) / len(sft_pts)
