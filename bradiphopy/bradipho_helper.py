# -*- coding: utf-8 -*-

import numpy as np
import vtk.util.numpy_support as ns
import vtk

from bradiphopy.utils import numpy_to_vtk_array


class BraDiPhoHelper3D():
    """Helper class for VTK objects."""

    def __init__(self, polydata):
        new_polydata = vtk.vtkPolyData()
        new_polydata.DeepCopy(polydata)
        self.polydata = new_polydata

    def __str__(self):
        """ Generate the string for printing """

        txt = "BraDiPhoHelper3D object with {} points and {} cells.\n".format(
            self.polydata.GetNumberOfPoints(), self.polydata.GetNumberOfCells())
        txt += "Scalars: {}\n".format(self.get_scalar_names())

        return txt

    def __len__(self):
        return self.polydata.GetNumberOfPoints()

    @staticmethod
    def generate_bdp_obj(vertices, triangles=[], arrays=None):
        polydata = vtk.vtkPolyData()
        if len(triangles):
            vtk_triangles = np.hstack(np.c_[np.ones(len(triangles),
                                                    dtype=int) * 3,
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
        """Subsample the vertices of a polydata object."""
        if not len(indices):
            return BraDiPhoHelper3D(vtk.vtkPolyData())
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
        if len(array) != self.polydata.GetNumberOfPoints():
            raise ValueError("Array length does not match number of points.")
        vtk_array = numpy_to_vtk_array(np.array(array), name=name, dtype=dtype)

        if 'normal' in name.lower():
            self.polydata.GetPointData().SetNormals(vtk_array)
        else:
            try:
                self.get_scalar(name)
                self.polydata.GetPointData().SetScalars(vtk_array)
            except ValueError:
                self.polydata.GetPointData().AddArray(vtk_array)

    def get_field_data(self, name):
        return ns.vtk_to_numpy(self.polydata.GetFieldData().GetArray(name))

    def set_field_data(self, array, name):
        vtk_array = numpy_to_vtk_array(np.array(array), name=name)
        self.polydata.GetFieldData().AddArray(vtk_array)

    def get_polydata_triangles(self):
        vtk_polys = ns.vtk_to_numpy(self.polydata.GetPolys().GetData())
        if len(vtk_polys) == 0:
            nbr_cells = self.polydata.GetNumberOfCells()
            triangles = []
            for i in range(nbr_cells):
                #     cell_ids = vtk.vtkIdList()
                ids = self.polydata.GetCell(i).GetPointIds()
                for j in range(ids.GetNumberOfIds()-2):
                    triangles.append([ids.GetId(j),
                                     ids.GetId(j+1),
                                     ids.GetId(j+2)])
            return np.array(triangles)
        else:
            if not (vtk_polys[::4] == 3).all():
                raise ValueError("Not all polygons are triangles")
        return np.vstack([vtk_polys[1::4], vtk_polys[2::4], vtk_polys[3::4]]).T

    def get_polydata_vertices(self):
        return ns.vtk_to_numpy(self.polydata.GetPoints().GetData())

    def set_polydata_vertices(self, vertices):
        vtk_points = vtk.vtkPoints()
        vtk_points.SetData(ns.numpy_to_vtk(vertices, deep=True))
        self.get_polydata().SetPoints(vtk_points)

    def set_polydata_triangles(self, triangles):
        vtk_triangles = np.hstack(
            np.c_[np.ones(len(triangles)).astype(np.int) * 3, triangles])
        vtk_triangles = ns.numpy_to_vtkIdTypeArray(vtk_triangles, deep=True)
        vtk_cells = vtk.vtkCellArray()
        vtk_cells.SetCells(len(triangles), vtk_triangles)
        self.get_polydata().SetPolys(vtk_cells)
