# -*- coding: utf-8 -*-

"""
"""

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from scipy.ndimage import gaussian_filter
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


def create_mesh_from_image(img, threshold=0.9, spacing=(1.0, 1.0, 1.0)):
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
    data = np.rot90(data, k=3, axes=(0, 2))
    data = np.flip(data, axis=1)
    affine = img.affine
    vox_sizes = img.header.get_zooms()[0:3]

    percentiles = np.percentile(data[data > 0], [1, 99])
    data[data < percentiles[0]] = 0
    data[data > percentiles[1]] = 0
    data = gaussian_filter(data, sigma=0.5)

    # Convert numpy array to VTK image data
    vtk_data = vtk.util.numpy_support.numpy_to_vtk(num_array=data.ravel(),
                                                   deep=True,
                                                   array_type=vtk.VTK_UNSIGNED_CHAR)
    image = vtk.vtkImageData()

    # Set the spacing, origin, and direction from the affine matrix
    offsets = (2 * affine[1, 3]) + (data.shape[1] * vox_sizes[1])
    image.SetSpacing(vox_sizes)

    image.SetOrigin(affine[:3, 3] - np.array([0, offsets-1, 0]))
    direction_matrix = vtk.vtkMatrix3x3()
    for i in range(3):
        for j in range(3):
            direction_matrix.SetElement(i, j, affine[i, j])
    image.SetDirectionMatrix(direction_matrix)

    image.SetDimensions(data.shape)
    image.GetPointData().SetScalars(vtk_data)

    # Apply marching cubes algorithm
    marching_cubes = vtk.vtkMarchingCubes()
    marching_cubes.SetInputData(image)
    marching_cubes.SetValue(0, threshold)
    marching_cubes.Update()

    # Apply a smoothing filter
    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputData(marching_cubes.GetOutput())
    smoother.SetNumberOfIterations(2)
    smoother.SetPassBand(0.1)
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()
    smoother.Update()

    # Return the transformed vtkPolyData
    return smoother.GetOutput()


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
    sampled_points = vtk.util.numpy_support.vtk_to_numpy(sampled_points_vtk.GetData())
    print(len(sampled_points))
    return sampled_points