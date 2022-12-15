# -*- coding: utf-8 -*-

import numpy as np
import vtk.util.numpy_support as ns
import vtk

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


class BraDiPhoHelper3D():
    """Helper class for VTK objects."""

    def __init__(self, polydata):
        self.polydata = polydata

    def __str__(self):
        """ Generate the string for printing """

        txt = "BraDiPhoHelper3D object with {} points and {} cells.\n".format(
            self.polydata.GetNumberOfPoints(), self.polydata.GetNumberOfCells())
        txt += "Scalars: {}\n".format(self.get_scalar_names())

        return txt

    @staticmethod
    def generate_bdp_obj(vertices, triangles=[], arrays=None):
        polydata = vtk.vtkPolyData()
        if len(triangles):
            vtk_triangles = np.hstack(np.c_[np.ones(len(triangles), dtype=int) * 3,
                                            triangles])
        else:
            vtk_triangles = np.array([], dtype=int)
        vtk_triangles = ns.numpy_to_vtkIdTypeArray(vtk_triangles, deep=True)

        vtk_cells = vtk.vtkCellArray()
        vtk_cells.SetCells(len(triangles), vtk_triangles)
        polydata.SetPolys(vtk_cells)

        vtk_points = vtk.vtkPoints()
        vtk_points.SetData(ns.numpy_to_vtk(vertices, deep=True))
        polydata.SetPoints(vtk_points)

        tmp_bdp_obj = BraDiPhoHelper3D(polydata)
        if arrays is not None:
            if isinstance(arrays, tuple):
                arrays = zip(arrays)

            if isinstance(arrays, zip):
                arrays = dict(arrays)

            for name, array in arrays.items():
                tmp_bdp_obj.set_scalar(array, name)

        return tmp_bdp_obj

    def get_polydata(self):
        return self.polydata

    def set_polydata(self, polydata):
        self.polydata = polydata

    def get_scalar_names(self):
        names = []
        i = 0
        while True:
            curr_name = self.polydata.GetPointData().GetArrayName(i)
            if curr_name is None:
                break
            names.append(curr_name)
            i += 1
        return names

    def subsample_polydata_vertices(self, indices):
        if np.min(indices) < 0 or \
            np.max(indices) >= self.polydata.GetNumberOfPoints():
            raise ValueError("Indices out of range.")

        if self.get_polydata_triangles() == []:
            return self.subsample_mesh_vertices(indices)
        else:
            return self.subsample_point_cloud_vertices(indices)

    def subsample_mesh_vertices(self, indices):
        raise NotImplementedError

    def subsample_point_cloud_vertices(self, indices):
        vertices = self.get_polydata_vertices()
        names = self.get_scalar_names()
        arrays = [self.get_scalar(name) for name in names]

        new_vertices = vertices[indices]
        new_arrays = [array[indices] for array in arrays]

        polydata = self.generate_bdp_obj(new_vertices,
                                          arrays=zip(names, new_arrays))
        return polydata

    def get_scalar(self, name):
        scalar = self.polydata.GetPointData().GetScalars(name)
        if scalar is None:
            raise ValueError("No scalar named '{}'".format(name))

        return ns.vtk_to_numpy(scalar)

    def set_scalar(self, array, name, dtype=None):
        if dtype is not None:
            vtk_dtype = datatype_map[np.dtype(dtype)]
        else:
            vtk_dtype = datatype_map[np.dtype(array.dtype)]
        vtk_array = ns.numpy_to_vtk(array, deep=True,
                                    array_type=vtk_dtype)
        vtk_array.SetName(name)

        if name == 'Normals':
            self.polydata.GetPointData().SetNormals(vtk_array)
        else:
            self.polydata.GetPointData().SetScalars(vtk_array)

    def get_polydata_triangles(self):
        vtk_polys = ns.vtk_to_numpy(self.polydata.GetPolys().GetData())
        if not (vtk_polys[::4] == 3).all():
            raise ValueError("Not all polygons are triangles")
        return np.vstack([vtk_polys[1::4], vtk_polys[2::4], vtk_polys[3::4]]).T

    def get_polydata_vertices(self):
        return ns.vtk_to_numpy(self.polydata.GetPoints().GetData())
