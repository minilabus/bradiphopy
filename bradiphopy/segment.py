
import numpy as np
from scilpy.tractograms.streamline_operations import generate_matched_points

from scipy.spatial import KDTree
from scilpy.tractograms.streamline_operations import resample_streamlines_num_points

def filter_from_surface(sft, bdp_obj, mode, criteria, distance):
    """
    Filter streamlines based on their proximity to a surface.
    """
    # matched_points = generate_matched_points(sft)
    sft_pts = sft.streamlines._data
    surf_pts = bdp_obj.get_polydata_vertices()

    tree = KDTree(surf_pts)
    dist, pts_ind = tree.query(sft_pts, k=1, distance_upper_bound=distance)

    indices = []
    tmp_len = [len(s) for s in sft.streamlines]
    offsets = np.insert(np.cumsum(tmp_len), 0, 0)

    for i in range(len(offsets) - 1):
        curr_distance = dist[offsets[i]:offsets[i+1]]
        if mode == 'any' and np.any(curr_distance != np.inf):
            indices.append(i)
        elif mode == 'all' and np.all(curr_distance != np.inf):
            indices.append(i)
        elif mode == 'either_end' and (curr_distance[0] != np.inf or \
                                        curr_distance[-1] != np.inf):
                indices.append(i)
        elif mode == 'both_ends' and (curr_distance[0] != np.inf and \
                                        curr_distance[-1] != np.inf):
            indices.append(i)

    if criteria == 'exclude':
        indices = np.setdiff1d(np.arange(len(sft)), indices)

    return indices

def get_proximity_scores(sft, bdp_obj, distance=1, endpoints_only=False):
    """
    Get the proximity scores of the streamlines to a surface.
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
