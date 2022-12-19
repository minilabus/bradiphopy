#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to rotate a scene around a mesh (not point cloud)
"""

import math
import shutil
import argparse
import os

from fury import window, utils
import numpy as np

from bradiphopy.bradipho_helper import BraDiPhoHelper3D
from bradiphopy.io import load_polydata
from bradiphopy.fury import record


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_file', nargs='+',
                   help='Input filename.')
    p.add_argument('--out_dir',
                   help='Output filename.')

    p.add_argument('-f', dest='overwrite', action='store_true',
                   help='Force overwriting of the output files.')
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if os.path.isdir(args.out_dir):
        if not args.overwrite:
            raise ValueError(
                'Output directory already exists. Use -f to force overwrite.')
        else:
            shutil.rmtree(args.out_dir)
            os.mkdir(args.out_dir)
    else:
        os.mkdir(args.out_dir)

    polydata = load_polydata(args.in_file[0])
    bdp_obj = BraDiPhoHelper3D(polydata)
    if bdp_obj.get_polydata_triangles().size == 0:
        raise ValueError('Input mesh is not a triangle mesh (point clound?).')

    actor = utils.get_actor_from_polydata(polydata)
    scene = window.Scene()
    scene.reset_camera()
    scene.add(actor)

    focal_point = scene.get_camera()[1]
    view_up = (0, 0, 1)
    r = np.linalg.norm(scene.get_camera()[0])
    n = int(360/args.revolution_increment)
    circum_pts = [(math.cos(2*math.pi/n*x)*r, math.sin(2*math.pi/n*x)*r, 0)
                  for x in range(0, n)]
    circum_pts = np.array(circum_pts)

    scene.set_camera(position=circum_pts[0],
                     focal_point=focal_point,
                     view_up=view_up)

    nbr_actor = len(args.in_file)
    when_to_switch = np.ceil(len(circum_pts) / nbr_actor)

    count = 0
    to_switch = True
    for i in range(len(circum_pts)):
        if to_switch:
            polydata = load_polydata(args.in_file[count])
            actor = utils.get_actor_from_polydata(polydata)
            scene.add(actor)
            to_switch = False
        else:
            scene.add(actor)
        if i % when_to_switch == 0 and count+1 < nbr_actor and i > 1:
            to_switch = True
            count += 1

        scene.set_camera(position=circum_pts[i],
                         focal_point=focal_point,
                         view_up=view_up)
        record(scene, size=(600, 600),
               out_path=os.path.join(args.out_dir, 'f_{:03d}.png'.format(i)),
               position=circum_pts[i], focal_point=focal_point, view_up=view_up)

        scene.rm_all()


if __name__ == '__main__':
    main()
