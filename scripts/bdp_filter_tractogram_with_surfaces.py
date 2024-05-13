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
"""

import argparse

from dipy.io.streamline import load_tractogram, save_tractogram
import numpy as np

from bradiphopy.bradipho_helper import BraDiPhoHelper3D
from bradiphopy.io import load_polydata
from bradiphopy.segment import filter_from_surface

MODES = ['any', 'all', 'either_end', 'both_ends']
CRITERIA = ['include', 'exclude']


def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)

    p.add_argument('in_tractogram',
                   help='Path of the input tractogram file.')
    p.add_argument('out_tractogram',
                   help='Path of the output tractogram file.')

    p.add_argument('--individual_surface', nargs='+', action='append',
                   help="ROI_NAME MODE CRITERIA DISTANCE "
                        "(distance in mm is optional)\n"
                        "Filename of a surface to use as a ROI.")

    return p



def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    
    min_arg = 0 if args.individual_surface is None else len(args.individual_surface)
    if min_arg == 0:
        raise ValueError("At least one ROI must be provided.")
    
    sft = load_tractogram(args.in_tractogram, 'same')
    indices = np.arange(len(sft.streamlines))
    for surf_opt in args.individual_surface:
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
            raise ValueError(f"Criteria {criteria} is not valid. Use one of {CRITERIA}.")

        polydata = load_polydata(roi_name, to_lps=True)
        bdp_obj = BraDiPhoHelper3D(polydata)
        curr_indices = filter_from_surface(sft, bdp_obj, mode,
                                           criteria, distance)
        indices = np.intersect1d(indices, curr_indices)
        sft = sft[indices]

    save_tractogram(sft, args.out_tractogram, bbox_valid_check=False)



if __name__ == "__main__":
    main()
