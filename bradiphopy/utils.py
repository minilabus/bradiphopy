# -*- coding: utf-8 -*-

"""
"""

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from scipy.ndimage import gaussian_filter, binary_dilation
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


def get_colormap(name):
    """Get a matplotlib colormap from a name or a list of named colors.
    Parameters
    ----------
    name : str
        Name of the colormap or a list of named colors (separated by a -).
    Returns
    -------
    matplotlib.colors.Colormap
        The colormap
    """

    if '-' in name:
        name_list = name.split('-')
        colors_list = [colors.to_rgba(color)[0:3] for color in name_list]
        cmap = colors.LinearSegmentedColormap.from_list('CustomCmap',
                                                        colors_list)
        return cmap

    return plt.cm.get_cmap(name)


def create_mesh_from_image(img, dilate=0, threshold=0.5):
    """
    Generate a mesh using the marching cubes algorithm from an image.

    Parameters:
    coords (numpy.ndarray): Array (to be binarized).
    threshold (float): Threshold for the marching cubes algorithm.
    spacing (tuple): Spacing for the vtkImageData in mm.

    Returns:
    vtkPolyData: Generated mesh as vtkPolyData.
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
    vtk_data = vtk.util.numpy_support.numpy_to_vtk(num_array=data.ravel(order='F'),
                                                   deep=True,
                                                   array_type=vtk.VTK_DOUBLE)
    image = vtk.vtkImageData()

    # Set the spacing, origin, and direction from the affine matrix
    image.SetSpacing(VOX_SIZE)

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
    Sample a given number of points from the surface of a mesh to create a point cloud.

    Parameters:
    mesh (vtkPolyData): The mesh from which to sample points.
    sampling_distance (float): The distance between two sampled points.

    Returns:
    numpy.ndarray: An array of sampled points forming the point cloud.
    """
    # Create a filter to sample the surface
    surface_filter = vtk.vtkPolyDataPointSampler()
    surface_filter.SetInputData(mesh)
    surface_filter.SetDistance(sampling_distance)
    surface_filter.Update()

    # Extract points from the filter
    sampled_points_vtk = surface_filter.GetOutput().GetPoints()
    sampled_points = vtk.util.numpy_support.vtk_to_numpy(
        sampled_points_vtk.GetData())

    return sampled_points
