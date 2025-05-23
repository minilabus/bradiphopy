# -*- coding: utf-8 -*-
"""
Provides modified or fixed versions of functions originally from the FURY
library for 3D visualization tasks, such as scene recording and polyline
manipulation.
"""

from fury.colormap import line_colors
from fury.io import save_image
from fury.lib import (RenderWindow, RenderLargeImage)
from fury.utils import (map_coordinates_3d_4d,
                        numpy_to_vtk_cells,
                        numpy_to_vtk_points)
from fury.window import Scene
import numpy as np
import vtk
import vtk.util.numpy_support as ns

from bradiphopy.utils import numpy_to_vtk_array


def record(scene=None, out_path=None, size=(300, 300),
           position=(0, 0, 1), focal_point=(0, 0, 0), view_up=(0, 0, 1)):
    """
    Records a single frame from a FURY scene or VTK renderer to an image file.

    Parameters
    ----------
    scene : fury.window.Scene or vtk.vtkRenderer, optional
        The scene or renderer to record. If None, a new `Scene` is created.
        Defaults to None.
    out_path : str, optional
        Path to save the output image. If None, defaults to "fury.png" in the
        current working directory. Defaults to None.
    size : tuple of int, optional
        A 2-tuple `(width, height)` for the output image resolution.
        Defaults to `(300, 300)`.
    position : tuple of float, optional
        Camera position `(x, y, z)`. Defaults to `(0, 0, 1)`.
    focal_point : tuple of float, optional
        Camera focal point `(x, y, z)`. Defaults to `(0, 0, 0)`.
    view_up : tuple of float, optional
        Camera view-up vector `(x, y, z)`. Defaults to `(0, 0, 1)`.
    """
    if scene is None:
        scene = Scene()

    # Set camera parameters
    camera = scene.GetActiveCamera()
    if camera:
        camera.SetPosition(position)
        camera.SetFocalPoint(focal_point)
        camera.SetViewUp(view_up)
        scene.ResetCameraClippingRange()

    renWin = RenderWindow()
    renWin.SetOffScreenRendering(1)
    renWin.SetBorders(False)
    renWin.AddRenderer(scene)
    renWin.SetSize(size[0], size[1])

    scene.ResetCamera()

    renderLarge = RenderLargeImage()
    renderLarge.SetInput(scene)
    renderLarge.SetMagnification(1)
    renderLarge.Update()

    arr = ns.vtk_to_numpy(renderLarge.GetOutput().GetPointData()
                          .GetScalars())
    h, w, _ = renderLarge.GetOutput().GetDimensions() # Corrected order: VTK gives H, W, Dims
    components = renderLarge.GetOutput().GetNumberOfScalarComponents()
    arr = arr.reshape((h, w, components))
    if out_path is None:
        out_path = "fury.png"
    save_image(arr, out_path)

    renWin.RemoveRenderer(scene)
    renWin.Finalize()


def lines_to_vtk_polydata(lines, colors=None):
    """
    Creates a `vtk.vtkPolyData` object from a list of lines (polylines) and
    applies specified colors to its points.

    Parameters
    ----------
    lines : list of numpy.ndarray
        A list where each element is a NumPy array of shape (M_i, 3),
        representing the M_i points of the i-th line.
    colors : array_like, optional
        Coloring scheme for the lines. The interpretation depends on the
        shape and type of `colors`:
        - If None or False: A standard orientation-based colormap is applied
          to each line.
        - tuple (3,): A single RGB color applied to all lines.
        - numpy.ndarray (N, 3): N is the number of lines. Each line gets a
          distinct RGB color.
        - list of numpy.ndarray: Each array in the list corresponds to a line
          and specifies RGB colors for each point in that line.
        - numpy.ndarray (K, 3): K is the total number of points across all
          lines. Each point gets a distinct RGB color.
        - numpy.ndarray (K,): K is the total number of points. These values
          are used for colormap lookup for each point.
        - numpy.ndarray (L,): L is the number of lines. These values are
          used for colormap lookup, applied per line.
        - numpy.ndarray (X, Y, Z) or (X, Y, Z, 3): Volumetric data used for
          trilinear interpolation of colors at line point coordinates.
        Defaults to None.

    Returns
    -------
    vtk.vtkPolyData
        Polydata object containing the lines with associated point data for
        colors (named "RGB").

    Raises
    ------
    ValueError
        If `lines` is empty.
    """
    # Get the 3d points_array
    if lines.__class__.__name__ == 'ArraySequence': # Support for DIPY's ArraySequence
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

        vtk_colors.SetName("RGB")
        polydata.GetPointData().SetScalars(vtk_colors)

    return polydata


def get_polydata_lines(line_polydata):
    """
    Extracts lines from a `vtk.vtkPolyData` object into a list of NumPy arrays.

    Parameters
    ----------
    line_polydata : vtk.vtkPolyData
        The input polydata object from which to extract lines. It is assumed
        that this polydata contains lines (polylines) in its cell structure.

    Returns
    -------
    list of numpy.ndarray
        A list where each element is an Ndarray of shape (M_i, 3),
        representing the M_i vertices of the i-th extracted line.
    """
    lines_vertices = ns.vtk_to_numpy(line_polydata.GetPoints().GetData())
    lines_idx = ns.vtk_to_numpy(line_polydata.GetLines().GetData())

    lines = []
    current_idx = 0
    while current_idx < len(lines_idx):
        line_len = lines_idx[current_idx]

        next_idx = current_idx + line_len + 1
        line_range = lines_idx[current_idx + 1: next_idx]

        lines += [lines_vertices[line_range]]
        current_idx = next_idx
    return lines
