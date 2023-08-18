import vtk
import sys
import numpy as np
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

# reader = vtk.vtkPLYReader()
reader = vtk.vtkXMLPolyDataReader()
reader.SetFileName(sys.argv[1])
reader.Update()

import vtk.util.numpy_support as ns


import numpy as np
arr = np.array([int(sys.argv[2])], dtype=int)
arr = numpy_to_vtk_array(np.array(arr), name='TimeValue')
polydata = reader.GetOutput()
# polydata.GetPointData().SetScalars(None) # Erase RGB

polydata.GetFieldData().AddArray(arr)
print(ns.vtk_to_numpy(polydata.GetFieldData().GetArray('TimeValue')))

writer = vtk.vtkXMLPolyDataWriter()
# writer.SetArrayName("RGB")
writer.SetFileName(sys.argv[3])
writer.SetInputData(polydata)
writer.Update()
writer.Write()