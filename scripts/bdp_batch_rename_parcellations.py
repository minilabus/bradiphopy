#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

"""

import argparse
import json
import os
import shutil


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_dir',
                   help='Input directory.')
    p.add_argument('in_json',
                   help='Input JSON file.')
    p.add_argument('out_dir',
                   help='Output directory.')
    p.add_argument('--extention', default='ply',
                   help='File extention to use[%(default)s].')
    p.add_argument('-f', dest='overwrite', action='store_true',
                   help='Force overwriting of the output files.')
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if not os.path.isdir(args.in_dir):
        raise IOError(
            '{} does not exist.'.format(args.in_dir))
    if not os.path.isfile(args.in_json):
        raise IOError(
            '{} does not exist.'.format(args.in_json))

    if not os.path.isdir(args.out_dir):
        os.mkdir(args.out_dir)
    else:
        if not args.overwrite:
            raise IOError(
                '{} already exists, use -f to overwrite.'.format(args.out_dir))
        if os.listdir(args.out_dir):
            shutil.rmtree(args.out_dir)
            os.mkdir(args.out_dir)

    with open(args.in_json, 'r') as f:
        data = json.load(f)

    ext = args.extention if args.extention.startswith('.') else '.' + args.extention

    # Check which files are in the input directory (key vs value)
    total_entries = len(data)
    key_count, value_count = 0, 0
    for key, value in data.items():
        key, value = str(key), str(value)
        in_file_key = os.path.join(args.in_dir, f'{key}{ext}')
        in_file_value = os.path.join(args.in_dir, f'{value}{ext}')

        if os.path.isfile(in_file_key):
            key_count += 1
        if os.path.isfile(in_file_value):
            value_count += 1

    if value_count == key_count:
        raise ValueError('Both keys and values are in the input directory.\n'
                         'The input folder must have either naming convention, '
                         'not both.')
    if total_entries - key_count < total_entries - value_count:
        use_key = True
        print('Using the keys as input file.')
    else:
        use_key = False
        print('Using the values as input file.')

    for key, value in data.items():
        key, value = str(key), str(value)
        if use_key:
            in_file = os.path.join(args.in_dir, f'{key}{ext}')
            out_file = os.path.join(args.out_dir, f'{value}{ext}')
        else:
            in_file = os.path.join(args.in_dir, f'{value}{ext}')
            out_file = os.path.join(args.out_dir, f'{key}{ext}')

        if not os.path.isfile(in_file):
            continue

        if os.path.isfile(out_file) and not args.overwrite:
            raise IOError(
                '{} already exists, use -f to overwrite.'.format(out_file))

        shutil.copyfile(in_file, out_file)


if __name__ == '__main__':
    main()
