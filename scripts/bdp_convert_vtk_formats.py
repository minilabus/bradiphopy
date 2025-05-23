#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert PolyData from and to any of these extensions:
    [".vtk", ".vtp", ".ply", ".stl", ".xml", ".obj"]

MI-Brain does not support coloring of surfaces (when loading).
"""

import argparse
import os

from bradiphopy.io import load_polydata, save_polydata


def _build_arg_parser():
    """Builds and returns an argparse.ArgumentParser for this script.

    The parser is configured with arguments for:
    - Input file (VTK-supported format).
    - Output file (VTK-supported format).
    - A flag for saving the output file in ASCII format.
    - An overwrite flag for existing output files.
    The script's module-level docstring is used as the description.

    Returns:
        argparse.ArgumentParser: The configured argument parser.
    """
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_file',
                   help='Input filename (must be supported by VTK).')
    p.add_argument('out_file',
                   help='Output filename (must be supported by VTK).')

    p.add_argument('--ascii', action='store_true',
                   help='Save the file with data as ASCII '
                        '(instead of binary).')
    p.add_argument('-f', dest='overwrite', action='store_true',
                   help='Force overwriting of the output files.')
    return p


def main():
    """Main function to convert between VTK-supported polydata formats.

    Parses command-line arguments. Loads the polydata from the input file
    using `bradiphopy.io.load_polydata`. Then, saves the loaded polydata
    to the specified output file format using `bradiphopy.io.save_polydata`,
    optionally in ASCII format.
    """
    parser = _build_arg_parser()
    args = parser.parse_args()

    if os.path.isfile(args.out_file) and not args.overwrite:
        raise IOError(
            '{} already exists, use -f to overwrite.'.format(args.out_file))

    polydata = load_polydata(args.in_file)
    save_polydata(polydata, args.out_file, ascii=args.ascii)


if __name__ == '__main__':
    main()
