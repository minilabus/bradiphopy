# -*- coding: utf-8 -*-

"""
"""

from fury.colormap import line_colors
from fury.utils import (map_coordinates_3d_4d,
                        numpy_to_vtk_cells,
                        numpy_to_vtk_points)
import numpy as np
import vtk
import vtk.util.numpy_support as ns

datatype_map = {
    np.dtype('int8'): vtk.VTK_CHAR,
    np.dtype('uint8'): vtk.VTK_UNSIGNED_CHAR,
    np.dtype('int16'): vtk.VTK_SHORT,
    np.dtype('uint16'): vtk.VTK_UNSIGNED_SHORT,
    np.dtype('int32'): vtk.VTK_INT,
    np.dtype('uint32'): vtk.VTK_UNSIGNED_INT,
    np.dtype('int64'): vtk.VTK_LONG_LONG,
    np.dtype('uint64'): vtk.VTK_UNSIGNED_LONG_LONG,
    np.dtype('float32'): vtk.VTK_FLOAT,
    np.dtype('float64'): vtk.VTK_DOUBLE,
}


def numpy_to_vtk_array(array, name=None, dtype=None, deep=True):
    if dtype is not None:
        vtk_dtype = datatype_map[np.dtype(dtype)]
    else:
        vtk_dtype = datatype_map[np.dtype(array.dtype)]
    vtk_array = ns.numpy_to_vtk(np.asarray(array), deep=True,
                                array_type=vtk_dtype)
    if name is not None:
        vtk_array.SetName(name)
    return vtk_array


def lines_to_vtk_polydata(lines, colors=None):
    """Create a vtkPolyData with lines and colors.
    Parameters
    ----------
    lines : list
        list of N curves represented as 2D ndarrays
    colors : array (N, 3), list of arrays, tuple (3,), array (K,)
        If None or False, a standard orientation colormap is used for every
        line.
        If one tuple of color is used. Then all streamlines will have the same
        colour.
        If an array (N, 3) is given, where N is equal to the number of lines.
        Then every line is coloured with a different RGB color.
        If a list of RGB arrays is given then every point of every line takes
        a different color.
        If an array (K, 3) is given, where K is the number of points of all
        lines then every point is colored with a different RGB color.
        If an array (K,) is given, where K is the number of points of all
        lines then these are considered as the values to be used by the
        colormap.
        If an array (L,) is given, where L is the number of streamlines then
        these are considered as the values to be used by the colormap per
        streamline.
        If an array (X, Y, Z) or (X, Y, Z, 3) is given then the values for the
        colormap are interpolated automatically using trilinear interpolation.
    Returns
    -------
    poly_data : vtkPolyData
    color_is_scalar : bool, true if the color array is a single scalar
        Scalar array could be used with a colormap lut
        None if no color was used
    """
    # Get the 3d points_array
    if lines.__class__.__name__ == 'ArraySequence':
        points_array = lines._data
    else:
        points_array = np.vstack(lines)

    if points_array.size == 0:
        raise ValueError("Empty lines/streamlines data.")

    # Set Points to vtk array format
    vtk_points = numpy_to_vtk_points(points_array)

    # Set Lines to vtk array format
    vtk_cell_array = numpy_to_vtk_cells(lines)

    # Create the poly_data
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(vtk_points)
    polydata.SetLines(vtk_cell_array)

    # Get colors_array (reformat to have colors for each points)
    #           - if/else tested and work in normal simple case
    nb_points = len(points_array)
    nb_lines = len(lines)
    lines_range = range(nb_lines)
    points_per_line = [len(lines[i]) for i in lines_range]
    points_per_line = np.array(points_per_line, np.intp)

    color_is_scalar = False
    if points_array.size:
        if colors is None or colors is False:
            # set automatic rgb colors
            cols_arr = line_colors(lines)
            colors_mapper = np.repeat(lines_range, points_per_line, axis=0)
            vtk_colors = numpy_to_vtk_array(255 * cols_arr[colors_mapper],
                                            dtype=np.uint8)
        else:
            cols_arr = np.asarray(colors)
            if cols_arr.dtype == object:  # colors is a list of colors
                vtk_colors = numpy_to_vtk_array(255 * np.vstack(colors),
                                                dtype=np.uint8)
            else:
                if len(cols_arr) == nb_points:
                    if cols_arr.ndim == 1:  # values for every point
                        vtk_colors = ns.numpy_to_vtk(cols_arr,
                                                     deep=True)
                        color_is_scalar = True
                    elif cols_arr.ndim == 2:  # map color to each point
                        vtk_colors = numpy_to_vtk_array(255 * cols_arr,
                                                        dtype=np.uint8)

                elif cols_arr.ndim == 1:
                    if len(cols_arr) == nb_lines:  # values for every streamline
                        cols_arrx = []
                        for (i, value) in enumerate(colors):
                            cols_arrx += lines[i].shape[0]*[value]
                        cols_arrx = np.array(cols_arrx)
                        vtk_colors = ns.numpy_to_vtk(cols_arrx,
                                                     deep=True)
                        color_is_scalar = True
                    else:  # the same colors for all points
                        vtk_colors = numpy_to_vtk_array(
                            np.tile(255 * cols_arr, (nb_points, 1)),
                            dtype=np.uint8)

                elif cols_arr.ndim == 2:  # map color to each line
                    colors_mapper = np.repeat(lines_range, points_per_line,
                                              axis=0)
                    vtk_colors = numpy_to_vtk_array(
                        255 * cols_arr[colors_mapper],
                        dtype=np.uint8)
                else:  # colormap
                    #  get colors for each vertex
                    cols_arr = map_coordinates_3d_4d(cols_arr, points_array)
                    vtk_colors = ns.numpy_to_vtk(cols_arr, deep=True)
                    color_is_scalar = True

        vtk_colors.SetName("RGB")
        polydata.GetPointData().SetScalars(vtk_colors)

    return polydata
