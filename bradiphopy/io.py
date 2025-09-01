# -*- coding: utf-8 -*-
"""
Functions to facilitate IO with surfaces and their additional data.
"""

import os
import vtk


VTK_EXTENSIONS = [".vtk", ".vtp", ".ply", ".stl", ".xml", ".obj"]


def load_polydata(filename, to_lps=False):
    """
    Loads polygonal data from a file into a `vtk.vtkPolyData` object.

    Parameters
    ----------
    filename : str
        Path to the input file. Supported extensions are: .vtk, .vtp, .fib,
        .ply, .stl, .xml, .obj.
    to_lps : bool, optional
        If True, applies a transformation to convert the data to LPS
        (Left Posterior Superior) coordinate system by flipping the X and Y
        axes. Defaults to False.

    Returns
    -------
    vtk.vtkPolyData
        The loaded polydata object, potentially transformed to LPS.

    Raises
    ------
    IOError
        If the file does not exist or if the file extension is not supported.
    """
    if not os.path.isfile(filename):
        raise IOError("{} does not exist.".format(filename))

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
        raise IOError("{} is not supported by VTK.".format(ext))

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
    """
    Saves a `vtk.vtkPolyData` object to a file.

    Parameters
    ----------
    polydata : vtk.vtkPolyData
        The polydata object to save.
    filename : str
        Path to the output file. Supported extensions are: .vtk, .vtp, .fib,
        .ply, .stl, .xml, .obj.
    ascii : bool, optional
        If True, saves in ASCII format. Otherwise, binary format is used where
        available (e.g., .ply, .stl). Defaults to True. For .ply, attempts to
        set color array name to "RGB" for some viewers.

    Returns
    -------
    None

    Raises
    ------
    IOError
        If the file extension is not supported.
    """
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
        raise IOError("{} is not supported by VTK.".format(ext))

    writer.SetFileName(filename)
    writer.SetInputData(polydata)

    if ascii:
        if hasattr(writer, "SetFileTypeToASCII"):
            writer.SetFileTypeToASCII()
    else:
        if hasattr(writer, "SetFileTypeToBinary"):
            writer.SetFileTypeToBinary()
        # For STL, binary is the default if FileType is not set to ASCII
        # For PLY, SetFileTypeToBinary is available.
        # For VTK legacy, SetFileTypeToBinary is available.
        # For VTP (XML), it's text based but can have compressed binary inline.
        # SetFileTypeToBinary is not typical for XML based writers like vtkXMLPolyDataWriter.
        # OBJ is typically text.

    writer.Update()
    writer.Write()
