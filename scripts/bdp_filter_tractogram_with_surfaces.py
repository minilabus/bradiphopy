#!/usr/bin/env python3
#  -*- coding: utf-8 -*-

"""
Filter a tractogram with surfaces. The streamlines are kept if they are close
to any of the surfaces.

The proximity is defined as the distance between a streamline and the closest
point on a surface. The distance is computed in mm.

The mode defines how the proximity is computed:
- any: keep the streamline if at least one point is close to the surface.
- all: keep the streamline if all points are close to the surface.
- either_end: keep the streamline if at least one of the endpoints is close to
  the surface.
- both_ends: keep the streamline if both endpoints are close to the surface.

The criteria defines if the streamlines close to the surface should be included
or excluded from the output tractogram.

--reuse_matched_pts can be used to reuse the matched points when filtering.
Since we are using a distance the same points can respect more than one
condition.
"""

import argparse
import os

from dipy.io.stateful_tractogram import StatefulTractogram
from dipy.io.streamline import load_tractogram, save_tractogram
import numpy as np

from bradiphopy.bradipho_helper import BraDiPhoHelper3D
from bradiphopy.io import load_polydata
from bradiphopy.segment import filter_from_surface

MODES = ['any', 'all', 'either_end', 'both_ends']
CRITERIA = ['include', 'exclude']


def _build_arg_parser():
    """Builds and returns an argparse.ArgumentParser for this script.

    The parser is configured with arguments for:
    - Input tractogram file.
    - Output tractogram file.
    - One or more '--individual_surface' arguments, each specifying
      a surface file, filtering mode, criteria, and an optional distance.
    - A flag '--reuse_matched_pts' to control how points matched by one
      surface are treated by subsequent surface filters.
    - An overwrite flag for existing output files.
    The script's module-level docstring is used as the description.

    Returns:
        argparse.ArgumentParser: The configured argument parser.
    """
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)

    p.add_argument('in_tractogram',
                   help='Path of the input tractogram file.')
    p.add_argument('out_tractogram',
                   help='Path of the output tractogram file.')

    p.add_argument('--individual_surface', nargs='+', action='append',
                   help='ROI_NAME MODE CRITERIA DISTANCE '
                        '(distance in mm is optional)\n'
                        'Filename of a surface to use as a ROI.')
    p.add_argument('--reuse_matched_pts', action='store_true',
                   help='Reuse already matched points when '
                        'filtering with multiple surfaces.')
    p.add_argument('-f', dest='overwrite', action='store_true',
                   help='Force overwriting of the output files.')
    return p


def main():
    """Main function to filter a tractogram using one or more surfaces.

    Parses command-line arguments. Loads the input tractogram.
    Initializes an array to keep track of points on streamlines that have
    been matched by a surface.
    Iterates through each surface filter specified via '--individual_surface'.
    For each surface:
        - Parses the ROI name, mode, criteria, and distance.
        - Loads the surface polydata.
        - Calls `bradiphopy.segment.filter_from_surface` to get indices
          of streamlines satisfying the current filter conditions and an
          updated list of matched points on those streamlines.
        - The set of streamlines to keep is updated by intersecting the
          current indices with those from previous filtering steps.
        - The main tractogram (`sft`) is updated with the filtered streamlines.
        - If `reuse_matched_pts` is False, the `matched_pts` array for
          the next iteration is based only on points from the currently
          kept streamlines that were matched by the current surface.
    Finally, the filtered tractogram is saved to the output file.
    If no streamlines remain after filtering, an empty tractogram is saved.
    """
    parser = _build_arg_parser()
    args = parser.parse_args()

    if os.path.isfile(args.out_tractogram) and not args.overwrite:
        raise ValueError(f"{args.out_tractogram} already exists. Use -f to "
                         "overwrite.")

    min_arg = 0 if args.individual_surface is None else len(
        args.individual_surface)
    if min_arg == 0:
        raise ValueError("At least one ROI must be provided.")

    sft = load_tractogram(args.in_tractogram, 'same')
    matched_pts = np.zeros(len(sft.streamlines._data), dtype=bool)

    for surf_opt in args.individual_surface:
        indices = np.arange(len(sft))
        if len(surf_opt) == 3:
            roi_name, mode, criteria = surf_opt[0:3]
            distance = 1.0
        elif len(surf_opt) == 4:
            roi_name, mode, criteria, _ = surf_opt[0:4]
            distance = float(surf_opt[3])
        else:
            raise ValueError("Individual surface must have 5 or 6 arguments.")
        if mode not in MODES:
            raise ValueError(f"Mode {mode} is not valid. Use one of {MODES}.")
        if criteria not in CRITERIA:
            raise ValueError(
                f"Criteria {criteria} is not valid. Use one of {CRITERIA}.")

        polydata = load_polydata(roi_name, to_lps=True)
        bdp_obj = BraDiPhoHelper3D(polydata)
        curr_indices, matched_pts = filter_from_surface(sft, bdp_obj, mode,
                                                        criteria, distance, matched_pts)
        indices = np.intersect1d(indices, curr_indices)

        if len(indices) != 0:
            # Use the ArraySequence slicing to keep the matched_points
            arr_seq = sft.streamlines.copy()
            if not args.reuse_matched_pts:
                arr_seq._data = matched_pts
                matched_pts = arr_seq[indices].copy()._data
            sft = sft[indices]

        else:
            sft = StatefulTractogram.from_sft([], sft)

    save_tractogram(sft, args.out_tractogram, bbox_valid_check=False)


if __name__ == "__main__":
    main()
