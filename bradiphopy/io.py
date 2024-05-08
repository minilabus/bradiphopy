# -*- coding: utf-8 -*-
"""
Functions to facilitate IO with surfaces and their additional data.
"""

import os
import vtk


VTK_EXTENSIONS = [".vtk", ".vtp", ".ply", ".stl", ".xml", ".obj"]


def load_polydata(filename, to_lps=False):
    if not os.path.isfile(filename):
        raise IOError('{} does not exist.'.format(filename))

    ext = os.path.splitext(filename)[-1].lower()

    if ext == ".vtk":
        reader = vtk.vtkPolyDataReader()
    elif ext == ".vtp":
        reader = vtk.vtkPolyDataReader()
    elif ext == ".fib":
        reader = vtk.vtkPolyDataReader()
    elif ext == ".ply":
        reader = vtk.vtkPLYReader()
    elif ext == ".stl":
        reader = vtk.vtkSTLReader()
    elif ext == ".xml":
        reader = vtk.vtkXMLPolyDataReader()
    elif ext == ".obj":
        reader = vtk.vtkOBJReader()
        reader.SetFileName(filename)
        reader.Update()
    else:
        raise IOError('{} is not supported by VTK.'.format(ext))

    reader.SetFileName(filename)
    reader.Update()

    transform = vtk.vtkTransform()
    flip_LPS = vtk.vtkMatrix4x4()
    flip_LPS.Identity()
    if to_lps:
        flip_LPS.SetElement(0, 0, -1)
        flip_LPS.SetElement(1, 1, -1)
        transform.Concatenate(flip_LPS)

    transformFilter = vtk.vtkTransformPolyDataFilter()
    transformFilter.SetInputData(reader.GetOutput())
    transformFilter.SetTransform(transform)
    transformFilter.Update()

    return transformFilter.GetOutput()


def save_polydata(polydata, filename, ascii=True):
    ext = os.path.splitext(filename)[-1].lower()

    if ext == ".vtk":
        writer = vtk.vtkPolyDataWriter()
    elif ext == ".vtp":
        writer = vtk.vtkPolyDataWriter()
    elif ext == ".fib":
        writer = vtk.vtkPolyDataWriter()
    elif ext == ".ply":
        writer = vtk.vtkPLYWriter()
        writer.SetArrayName("RGB")
    elif ext == ".stl":
        writer = vtk.vtkSTLWriter()
    elif ext == ".xml":
        writer = vtk.vtkXMLPolyDataWriter()
    elif ext == ".obj":
        writer = vtk.vtkOBJWriter()
    else:
        raise IOError('{} is not supported by VTK.'.format(ext))

    writer.SetFileName(filename)
    writer.SetInputData(polydata)

    if ascii:
        writer.SetFileTypeToASCII()
    writer.Update()
    writer.Write()
