# -*- coding: utf-8 -*-
"""
Provides utility functions for data conversions between NumPy and VTK,
colormap retrieval, and 3D mesh/point cloud generation and manipulation
from medical image data.
"""

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from scipy.ndimage import gaussian_filter, binary_dilation
import vtk
import vtk.util.numpy_support as ns


datatype_map = {
    np.dtype("int8"): vtk.VTK_CHAR,
    np.dtype("uint8"): vtk.VTK_UNSIGNED_CHAR,
    np.dtype("int16"): vtk.VTK_SHORT,
    np.dtype("uint16"): vtk.VTK_UNSIGNED_SHORT,
    np.dtype("int32"): vtk.VTK_INT,
    np.dtype("uint32"): vtk.VTK_UNSIGNED_INT,
    np.dtype("int64"): vtk.VTK_LONG_LONG,
    np.dtype("uint64"): vtk.VTK_UNSIGNED_LONG_LONG,
    np.dtype("float32"): vtk.VTK_FLOAT,
    np.dtype("float64"): vtk.VTK_DOUBLE,
}


def numpy_to_vtk_array(array, name=None, dtype=None, deep=True):
    """
    Converts a NumPy array to a VTK data array.

    Parameters
    ----------
    array : numpy.ndarray
        The NumPy array to convert.
    name : str, optional
        The name to assign to the VTK array. Defaults to None.
    dtype : numpy.dtype or str, optional
        The desired NumPy data type for the array before VTK conversion.
        If None, the array's current dtype is used. Defaults to None.
    deep : bool, optional
        If True, a deep copy of the NumPy array is made for VTK.
        This parameter is effectively fixed to True due to the underlying
        `vtk.util.numpy_support.numpy_to_vtk` call. Defaults to True.

    Returns
    -------
    vtk.vtkDataArray
        The converted VTK data array.
    """
    if dtype is not None:
        vtk_dtype = datatype_map[np.dtype(dtype)]
    else:
        vtk_dtype = datatype_map[np.dtype(array.dtype)]
    # The 'deep' parameter in ns.numpy_to_vtk is hardcoded to True in the call
    vtk_array = ns.numpy_to_vtk(np.asarray(array), deep=True, array_type=vtk_dtype)
    if name is not None:
        vtk_array.SetName(name)
    return vtk_array


def get_colormap(name):
    """
    Get a Matplotlib colormap from a name or a list of named colors.

    Parameters
    ----------
    name : str
        Name of the Matplotlib colormap (e.g., 'viridis', 'jet') or a
        string of hyphen-separated color names (e.g., 'red-yellow-blue')
        to create a custom `LinearSegmentedColormap`.

    Returns
    -------
    matplotlib.colors.Colormap
        The requested Matplotlib colormap instance.
    """
    if "-" in name:
        name_list = name.split("-")
        colors_list = [colors.to_rgba(color)[0:3] for color in name_list]
        cmap = colors.LinearSegmentedColormap.from_list("CustomCmap", colors_list)
        return cmap

    return plt.cm.get_cmap(name)


def create_mesh_from_image(img, dilate=0, threshold=0.5):
    """
    Generates a mesh using the marching cubes algorithm from an image.

    The generated mesh is transformed to the LPS coordinate system.

    Parameters
    ----------
    img : nibabel.Nifti1Image or similar
        Input medical image object (must have `get_fdata()`,
        `header.get_zooms()`, and `affine` attributes).
    dilate : int, optional
        Number of iterations for binary dilation of the thresholded image
        before Gaussian smoothing. Defaults to 0 (no dilation).
    threshold : float, optional
        Threshold value for the marching cubes algorithm to generate the
        isosurface from the smoothed, thresholded image data.
        Defaults to 0.5.

    Returns
    -------
    vtk.vtkPolyData
        The generated mesh as a VTK PolyData object, transformed to LPS
        coordinate system.
    """
    data = img.get_fdata()
    VOX_SIZE = img.header.get_zooms()
    AFFINE = img.affine

    percentiles = np.percentile(data[data > 0], [1, 99])
    data[data < percentiles[0]] = 0
    data[data > percentiles[1]] = 0
    data[data > 0] = 1
    if dilate > 0:
        data = binary_dilation(data, iterations=dilate)
    data = gaussian_filter(data, sigma=1)

    # Convert numpy array to VTK image data
    vtk_data = vtk.util.numpy_support.numpy_to_vtk(
        num_array=data.ravel(order="F"), deep=True, array_type=vtk.VTK_DOUBLE
    )
    image = vtk.vtkImageData()

    # Adjust for voxel centering
    voxel_half = np.array(VOX_SIZE) / 2.0
    adjusted_origin = AFFINE[:3, 3] + np.dot(AFFINE[:3, :3], voxel_half)
    image.SetOrigin(adjusted_origin)

    # Set the spatial transformation matrix
    direction_matrix = vtk.vtkMatrix3x3()
    for i in range(3):
        for j in range(3):
            direction_matrix.SetElement(i, j, AFFINE[i, j])
    image.SetDirectionMatrix(direction_matrix)

    image.SetDimensions(data.shape)
    image.GetPointData().SetScalars(vtk_data)

    # Apply marching cubes algorithm
    marching_cubes = vtk.vtkMarchingCubes()
    marching_cubes.SetInputData(image)
    marching_cubes.SetValue(0, threshold)
    marching_cubes.Update()

    transform = vtk.vtkTransform()
    flip_LPS = vtk.vtkMatrix4x4()
    flip_LPS.Identity()
    flip_LPS.SetElement(0, 0, -1)
    flip_LPS.SetElement(1, 1, -1)
    transform.Concatenate(flip_LPS)

    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetInputData(marching_cubes.GetOutput())
    transformFilter.SetTransform(transform)
    transformFilter.Update()

    return transformFilter.GetOutput()


def sample_mesh_to_point_cloud(mesh, sampling_distance):
    """
    Samples points from the surface of a mesh to create a point cloud.

    Parameters
    ----------
    mesh : vtk.vtkPolyData
        The input mesh from which to sample points.
    sampling_distance : float
        The desired minimum distance between sampled points on the surface.
        This is used by `vtkPolyDataPointSampler.SetDistance()`.

    Returns
    -------
    numpy.ndarray
        Array of shape (N, 3) with coordinates of sampled points.
    """
    # Create a filter to sample the surface
    surface_filter = vtk.vtkPolyDataPointSampler()
    surface_filter.SetInputData(mesh)
    surface_filter.SetDistance(sampling_distance)
    surface_filter.Update()

    # Extract points from the filter
    sampled_points_vtk = surface_filter.GetOutput().GetPoints()
    sampled_points = vtk.util.numpy_support.vtk_to_numpy(
        sampled_points_vtk.GetData()
    )

    return sampled_points
