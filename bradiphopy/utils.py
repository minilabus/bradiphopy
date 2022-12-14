# -*- coding: utf-8 -*-

import numpy as np
import vtk
import vtk.util.numpy_support as ns


def numpy_to_vtk_points(points):
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(ns.numpy_to_vtk(np.asarray(points), deep=True,
                                       array_type=vtk.VTK_FLOAT))
    return vtk_points
