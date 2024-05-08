#!/usr/bin/env python3
#  -*- coding: utf-8 -*-

"""

"""

import argparse
from bradiphopy.io import load_polydata, save_polydata
from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.io.stateful_tractogram import (set_sft_logger_level,
                                         StatefulTractogram, Space)
from bradiphopy.bradipho_helper import BraDiPhoHelper3D
import numpy as np
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
                        "(distance in voxel is optional)\n"
                        "Filename of a hand drawn ROI (.nii or .nii.gz).")
    p.add_argument('--entire_parcellation', nargs='+', action='append',
                   help="ROI_NAME ANNOT ID MODE CRITERIA DISTANCE "
                        "(distance in voxel is optional)\n"
                        "Filename of an atlas (.nii or .nii.gz).")

    return p



def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    
    min_arg = 0 if args.individual_surface is None else len(args.individual_surface)
    min_arg += 0 if args.entire_parcellation is None else len(args.entire_parcellation)
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
