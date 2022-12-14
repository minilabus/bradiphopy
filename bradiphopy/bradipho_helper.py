# -*- coding: utf-8 -*-

import numpy as np
import vtk.util.numpy_support as ns
import vtk

datatype_map = {
    np.int8: vtk.VTK_CHAR,
    np.uint8: vtk.VTK_UNSIGNED_CHAR,
    np.int16: vtk.VTK_SHORT,
    np.uint16: vtk.VTK_UNSIGNED_SHORT,
    np.int32: vtk.VTK_INT,
    np.uint32: vtk.VTK_UNSIGNED_INT,
    np.int64: vtk.VTK_LONG_LONG,
    np.uint64: vtk.VTK_UNSIGNED_LONG_LONG,
    np.float32: vtk.VTK_FLOAT,
    np.float64: vtk.VTK_DOUBLE,
}


class BraDiPhoHelper3D():
    """Helper class for VTK objects."""

    def __init__(self, polydata):
        self.polydata = polydata

    def get_polydata(self):
        return self.polydata

    def set_polydata(self, polydata):
        self.polydata = polydata

    def get_scalar(self, name):
        scalar = self.polydata.GetPointData().GetScalars(name)
        return ns.vtk_to_numpy(scalar)

    def set_scalar(self, array, name, dtype=np.float32):
        vtk_array = ns.numpy_to_vtk(np.asarray(array), deep=True,
                                    array_type=datatype_map[dtype])
        vtk_array.SetName(name)
        self.polydata.GetPointData().SetScalars(vtk_array)

    def get_polydata_triangles(self):
        vtk_polys = ns.vtk_to_numpy(self.polydata.GetPolys().GetData())
        if not (vtk_polys[::4] == 3).all():
            raise ValueError("Not all polygons are triangles")
        return np.vstack([vtk_polys[1::4], vtk_polys[2::4], vtk_polys[3::4]]).T

    def get_polydata_vertices(self):
        return ns.vtk_to_numpy(self.polydata.GetPoints().GetData())
