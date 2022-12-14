# -*- coding: utf-8 -*-
""" 
Functions to facilitate IO with surfaces and their additional data
"""

import os
import vtk.util.numpy_support
import vtk


VTK_EXTENSIONS = [".vtk", ".vtp", ".ply", ".stl", ".xml", ".obj"]


def load_polydata(filename):
    file_extension = os.path.splitext(filename)[-1].lower()

    if file_extension == ".vtk":
        reader = vtk.vtkPolyDataReader()
    elif file_extension == ".vtp":
        reader = vtk.vtkPolyDataReader()
    elif file_extension == ".fib":
        reader = vtk.vtkPolyDataReader()
    elif file_extension == ".ply":
        reader = vtk.vtkPLYReader()
    elif file_extension == ".stl":
        reader = vtk.vtkSTLReader()
    elif file_extension == ".xml":
        reader = vtk.vtkXMLPolyDataReader()
    elif file_extension == ".obj":
        reader = vtk.vtkOBJReader()
        reader.SetFileName(filename)
        reader.Update()
    else:
        raise IOError('{} is not supported by VTK.'.format(ext))

    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()


def save_polydata(polydata, filename, binary=False):
    file_extension = os.path.splitext(filename)[-1].lower()

    if file_extension == ".vtk":
        writer = vtk.vtkPolyDataWriter()
    elif file_extension == ".vtp":
        writer = vtk.vtkPolyDataWriter()
    elif file_extension == ".fib":
        writer = vtk.vtkPolyDataWriter()
    elif file_extension == ".ply":
        writer = vtk.vtkPLYWriter()
    elif file_extension == ".stl":
        writer = vtk.vtkSTLWriter()
    elif file_extension == ".xml":
        writer = vtk.vtkXMLPolyDataWriter()
    elif file_extension == ".obj":
    	writer = vtk.vtkOBJWriter()
    else:
        raise IOError('{} is not supported by VTK.'.format(ext))

    writer.SetFileName(filename)
    writer.SetInputData(polydata)

    if binary:
        writer.SetFileTypeToBinary()
    writer.Update()
    writer.Write()
