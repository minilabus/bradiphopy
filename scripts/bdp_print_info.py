#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prints information related to surfaces in the format:
    [".vtk", ".vtp", ".ply", ".stl", ".xml", ".obj"]

Script prints amount of vertices, triangles and its bounding box
"""

import argparse
import numpy as np

from bradiphopy.bradipho_helper import BraDiPhoHelper3D
from bradiphopy.io import load_polydata


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument("in_file", help="Input filename (must be supported by VTK).")
    return p

def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    polydata = load_polydata(args.in_file)
    bdp_obj = BraDiPhoHelper3D(polydata)
    print(f"Number of Points : {bdp_obj.polydata.GetNumberOfPoints()}")
    print(f"Number of Triangles : {bdp_obj.polydata.GetNumberOfCells()}")

    BBOX = bdp_obj.get_bound()

    print(f"Bounding Box (xmin,xmax,ymin,ymax,zmin,zmax) : {tuple(BBOX)}")


if __name__ == "__main__":
    main()
