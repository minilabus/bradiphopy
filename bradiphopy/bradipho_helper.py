# -*- coding: utf-8 -*-
"""
Provides the `BraDiPhoHelper3D` class, a wrapper around VTK's `vtkPolyData`
to simplify common 3D data manipulations, including vertex and face operations,
scalar data handling, and subsampling.
"""

import numpy as np
import vtk.util.numpy_support as ns
import vtk

from bradiphopy.utils import numpy_to_vtk_array


class BraDiPhoHelper3D():
    """
    A helper class to manage and manipulate 3D geometric data using `vtkPolyData`.

    This class provides convenient methods for accessing and modifying vertices,
    triangles (polygons/cells), scalar data, and field data associated with a
    3D model or point cloud. It also includes functionality for generating new
    objects and subsampling.
    """

    def __init__(self, polydata):
        """
        Initializes the BraDiPhoHelper3D object.

        Parameters
        ----------
        polydata : vtk.vtkPolyData
            The input VTK polydata object to wrap. A deep copy of this
            object is made.
        """
        new_polydata = vtk.vtkPolyData()
        new_polydata.DeepCopy(polydata)
        self.polydata = new_polydata

    def __str__(self):
        """Generate the string for printing."""
        txt = "BraDiPhoHelper3D object with {} points and {} cells.\n".format(
            self.polydata.GetNumberOfPoints(), self.polydata.GetNumberOfCells())
        txt += "Scalars: {}\n".format(self.get_scalar_names())

        return txt

    def __len__(self):
        """
        Returns the number of points (vertices) in the polydata.

        Returns
        -------
        int
            Number of points.
        """
        return self.polydata.GetNumberOfPoints()

    @staticmethod
    def generate_bdp_obj(vertices, triangles=[], arrays=None):
        """
        Generates a new BraDiPhoHelper3D object from vertices, triangles, and scalar arrays.

        Parameters
        ----------
        vertices : numpy.ndarray
            Array of vertex coordinates, shape (N, 3).
        triangles : numpy.ndarray or list, optional
            Array or list of triangle definitions, shape (M, 3), where each
            row contains indices of vertices forming a triangle. Defaults to
            an empty list (creating a point cloud).
        arrays : dict or zip, optional
            A dictionary or a zip object where keys are scalar names (str)
            and values are `numpy.ndarray` of scalar values corresponding
            to each vertex. Defaults to None.

        Returns
        -------
        BraDiPhoHelper3D
            A new instance of `BraDiPhoHelper3D` created from the provided data.
        """
        polydata = vtk.vtkPolyData()
        if len(triangles):
            vtk_triangles = np.hstack(np.c_[np.ones(len(triangles),
                                                    dtype=np.int64) * 3,  # Changed dtype to int64 for compatibility
                                            triangles])
        else:
            vtk_triangles = np.array([], dtype=np.int64) # Changed dtype to int64 for compatibility
        vtk_triangles = ns.numpy_to_vtkIdTypeArray(vtk_triangles, deep=True)

        vtk_cells = vtk.vtkCellArray()
        vtk_cells.SetCells(len(triangles) if len(triangles) > 0 else 0, vtk_triangles)
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
        """
        Returns the underlying `vtkPolyData` object.

        Returns
        -------
        vtk.vtkPolyData
            The VTK polydata object.
        """
        return self.polydata

    def set_polydata(self, polydata):
        """
        Sets the underlying `vtkPolyData` object.

        Parameters
        ----------
        polydata : vtk.vtkPolyData
            The VTK polydata object to set.
        """
        self.polydata = polydata

    def get_scalar_names(self):
        """
        Retrieves a list of names of all point data arrays (scalars).

        Returns
        -------
        list of str
            A list containing the names of the scalar arrays.
        """
        names = []
        i = 0
        while True:
            curr_name = self.polydata.GetPointData().GetArrayName(i)
            if curr_name is None:
                break
            names.append(curr_name)
            i += 1
        return names

    def subsample_polydata(self, indices):
        """
        Subsamples the polydata based on a list of vertex indices.

        Handles both meshes (with triangles) and point clouds. If triangles
        are present, it attempts to preserve them by calling `subsample_mesh`;
        otherwise, it calls `subsample_point_cloud_vertices`.

        Parameters
        ----------
        indices : list or numpy.ndarray
            List of integer indices of vertices to keep.

        Returns
        -------
        BraDiPhoHelper3D
            A new `BraDiPhoHelper3D` object containing the subsampled geometry.

        Raises
        ------
        ValueError
            If indices are out of range.
        """
        if not len(indices):
            return BraDiPhoHelper3D(vtk.vtkPolyData())
        if np.min(indices) < 0 or \
                np.max(indices) >= self.polydata.GetNumberOfPoints():
            raise ValueError("Indices out of range.")

        # Check if there are any polygons (triangles)
        if self.polydata.GetNumberOfPolys() > 0 or self.polydata.GetNumberOfStrips() > 0 :
            # Attempt to get triangles, if it fails (e.g. not all are triangles)
            # or returns an empty list for a valid reason other than no polys,
            # this logic might need refinement.
            # For now, assume get_polydata_triangles() is robust enough.
            try:
                triangles = self.get_polydata_triangles()
                if triangles.size > 0: # Check if triangles array is not empty
                    new_vertices, new_faces = self.subsample_mesh(indices)
                    # Preserve scalar data from the original object for the new vertices
                    original_scalars = {name: self.get_scalar(name)[indices] for name in self.get_scalar_names()}
                    bdo_obj = self.generate_bdp_obj(new_vertices, triangles=new_faces, arrays=original_scalars)
                    return bdo_obj
                else: # No triangles, treat as point cloud
                    return self.subsample_point_cloud_vertices(indices)
            except ValueError: # If get_polydata_triangles raises ValueError (e.g. not all polys are triangles)
                 return self.subsample_point_cloud_vertices(indices)

        else: # No polygons, definitely a point cloud
            return self.subsample_point_cloud_vertices(indices)

    def subsample_mesh(self, indices):
        """
        Subsamples a mesh (vertices and faces) based on a list of vertex indices.

        Only faces where all vertices are in the `indices` list will be kept,
        and vertex indices in faces will be remapped.

        Parameters
        ----------
        indices : list or numpy.ndarray
            List of integer indices of vertices to keep.

        Returns
        -------
        tuple
            - numpy.ndarray: The new vertex coordinates.
            - numpy.ndarray: The new triangle (face) definitions with remapped indices.
        """
        vertices = self.get_polydata_vertices()
        faces = self.get_polydata_triangles()
        indices = [ind for ind in indices if ind < len(vertices)]

        # return vertices[valid_indices], subsampled_faces
        indices = sorted(set(indices))

        # Filter the vertices using the indices list
        new_vertices = vertices[indices]

        # Create a mapping from old vertex indices to new vertex indices
        index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(indices)}

        # Iterate through each face, check if all vertices are in the new set, and remap indices
        mask = np.all(np.isin(faces, indices), axis=1)
        new_faces = []
        for i in np.where(mask)[0]:
            remapped_face = [index_map[idx] for idx in faces[i]]
            new_faces.append(remapped_face)

        return new_vertices, np.array(new_faces)

    def subsample_point_cloud_vertices(self, indices):
        """
        Subsamples a point cloud (vertices and associated scalar data)
        based on a list of vertex indices.

        Parameters
        ----------
        indices : list or numpy.ndarray
            List of integer indices of vertices to keep.

        Returns
        -------
        BraDiPhoHelper3D
            A new `BraDiPhoHelper3D` object containing the subsampled
            point cloud and its scalar data.
        """
        vertices = self.get_polydata_vertices()
        names = self.get_scalar_names()
        arrays = {name: self.get_scalar(name)[indices] for name in names}

        new_vertices = vertices[indices]

        bdo_obj = self.generate_bdp_obj(new_vertices, arrays=arrays)
        return bdo_obj

    def get_scalar(self, name):
        """
        Retrieves a specific point data array (scalar) by its name.

        Parameters
        ----------
        name : str
            The name of the scalar array to retrieve.

        Returns
        -------
        numpy.ndarray
            The scalar data as a NumPy array.

        Raises
        ------
        ValueError
            If no scalar with the given name exists.
        """
        scalar_array = self.polydata.GetPointData().GetArray(name) # Use GetArray for robust access
        if scalar_array is None:
            # Try GetScalars as a fallback, though GetArray is generally preferred
            scalar_array = self.polydata.GetPointData().GetScalars(name)
            if scalar_array is None:
                raise ValueError("No scalar named '{}'".format(name))

        return ns.vtk_to_numpy(scalar_array)

    def set_scalar(self, array, name, dtype=None):
        """
        Adds or updates a point data array (scalar).

        Parameters
        ----------
        array : numpy.ndarray
            The scalar data.
        name : str
            Name of the scalar.
        dtype : numpy.dtype, optional
            NumPy data type for the VTK array. Defaults to None (inferred).

        Raises
        ------
        ValueError
            If array length doesn't match the number of points.
        """
        if len(array) != self.polydata.GetNumberOfPoints():
            raise ValueError("Array length does not match number of points.")
        vtk_array = numpy_to_vtk_array(np.array(array), name=name, dtype=dtype)

        if 'normal' in name.lower(): # Case-insensitive check for 'normal'
            self.polydata.GetPointData().SetNormals(vtk_array)
        else:
            # Check if array already exists to replace it, otherwise add it
            if self.polydata.GetPointData().GetArray(name):
                self.polydata.GetPointData().RemoveArray(name) # Remove old before adding new with SetScalars or AddArray
            self.polydata.GetPointData().AddArray(vtk_array) # AddArray is safer for general scalars
            # Ensure it's set as active scalar if it's the only one or was intended to be
            if self.polydata.GetPointData().GetNumberOfArrays() == 1:
                 self.polydata.GetPointData().SetActiveScalars(name)


    def get_field_data(self, name):
        """
        Retrieves a specific field data array by its name.

        Parameters
        ----------
        name : str
            The name of the field data array.

        Returns
        -------
        numpy.ndarray
            The field data array.

        Raises
        ------
        ValueError
            If no field data with the given name exists.
        """
        field_array = self.polydata.GetFieldData().GetArray(name)
        if field_array is None:
            raise ValueError("No field data named '{}'".format(name))
        return ns.vtk_to_numpy(field_array)

    def set_field_data(self, array, name):
        """
        Adds a field data array.

        Parameters
        ----------
        array : numpy.ndarray
            The data to be added as a field array.
        name : str
            The name of the field data array.
        """
        vtk_array = numpy_to_vtk_array(np.array(array), name=name)
        self.polydata.GetFieldData().AddArray(vtk_array)

    def get_polydata_triangles(self):
        """
        Retrieves the triangle connectivity (faces) from the polydata as a NumPy array.

        Returns
        -------
        numpy.ndarray
            Array of shape (M, 3) representing triangle connectivity.
            Returns an empty array if no triangles are present or if cells
            are not uniformly triangles.

        Raises
        ------
        ValueError
            If polygons are present but not all are triangles.
        """
        if self.polydata.GetNumberOfPolys() == 0:
            return np.array([], dtype=int).reshape(0,3) # No polygons exist

        vtk_polys_data = self.polydata.GetPolys().GetData()
        if vtk_polys_data is None: # Should not happen if GetNumberOfPolys > 0
             return np.array([], dtype=int).reshape(0,3)

        vtk_polys = ns.vtk_to_numpy(vtk_polys_data)

        if len(vtk_polys) == 0: # No actual polygon data, though GetNumberOfPolys > 0
            # This case might indicate lines or verts stored as cells, but not polys.
            # Or, it could be an empty poly cell array.
            # Check generic cells if GetPolys is empty but GetNumberOfCells is not.
            if self.polydata.GetNumberOfCells() > 0 and self.polydata.GetNumberOfPolys() == 0:
                 # This implies cells are not vtkPoly, could be vtkTriangleStrip, etc.
                 # The original code had a fallback for this, let's try to adapt it.
                nbr_cells = self.polydata.GetNumberOfCells()
                triangles = []
                for i in range(nbr_cells):
                    cell = self.polydata.GetCell(i)
                    if cell.GetCellType() == vtk.VTK_TRIANGLE:
                        ids = cell.GetPointIds()
                        triangles.append([ids.GetId(0), ids.GetId(1), ids.GetId(2)])
                    # Add handling for other cell types if necessary, e.g., triangle strips
                    # For now, only explicit triangles are extracted here.
                if triangles:
                    return np.array(triangles, dtype=np.int64)


            return np.array([], dtype=np.int64).reshape(0, 3)


        # Assuming polys are triangles, each entry is count (3) followed by 3 indices.
        # Check if the counts are all 3.
        if not (vtk_polys[::4] == 3).all():
            # Fallback: Iterate through cells if GetPolys() is mixed or not all triangles
            # This part is tricky because GetPolys().GetData() flattens everything.
            # A more robust way is to iterate cells if the simple check fails.
            nbr_cells = self.polydata.GetNumberOfCells()
            triangles_from_cells = []
            has_non_triangle_poly = False
            for i in range(nbr_cells):
                cell = self.polydata.GetCell(i)
                if cell.GetCellType() == vtk.VTK_POLYGON: # Check if it's a polygon cell
                    ids = cell.GetPointIds()
                    if ids.GetNumberOfIds() == 3:
                        triangles_from_cells.append([ids.GetId(0), ids.GetId(1), ids.GetId(2)])
                    else:
                        has_non_triangle_poly = True # Found a polygon that is not a triangle
                        break # Stop if a non-triangle polygon is found
                elif cell.GetCellType() == vtk.VTK_TRIANGLE: # Also check for explicit triangle cells
                     ids = cell.GetPointIds()
                     triangles_from_cells.append([ids.GetId(0), ids.GetId(1), ids.GetId(2)])


            if has_non_triangle_poly:
                raise ValueError("Not all polygons are triangles.")
            if triangles_from_cells:
                return np.array(triangles_from_cells, dtype=np.int64)
            else: # No triangle cells found, or only non-polygonal cells
                return np.array([], dtype=np.int64).reshape(0,3)


        # If all counts are 3, then proceed with the fast extraction
        return np.vstack([vtk_polys[1::4], vtk_polys[2::4], vtk_polys[3::4]]).T.astype(np.int64)


    def get_polydata_vertices(self):
        """
        Retrieves the vertex coordinates from the polydata as a NumPy array.

        Returns
        -------
        numpy.ndarray
            Array of shape (N, 3) containing vertex coordinates.
        """
        return ns.vtk_to_numpy(self.polydata.GetPoints().GetData())

    def set_polydata_vertices(self, vertices):
        """
        Sets the vertex coordinates of the polydata.

        Parameters
        ----------
        vertices : numpy.ndarray
            Array of shape (N, 3) containing the new vertex coordinates.
        """
        vtk_points = vtk.vtkPoints()
        vtk_points.SetData(ns.numpy_to_vtk(vertices, deep=True))
        self.get_polydata().SetPoints(vtk_points)

    def set_polydata_triangles(self, triangles):
        """
        Sets the triangle connectivity (faces) of the polydata.

        Parameters
        ----------
        triangles : numpy.ndarray
            Array of shape (M, 3) containing triangle definitions.
        """
        vtk_triangles = np.hstack(
            np.c_[np.ones(len(triangles), dtype=np.int64) * 3, triangles.astype(np.int64)])
        vtk_triangles_array = ns.numpy_to_vtkIdTypeArray(vtk_triangles, deep=True)
        vtk_cells = vtk.vtkCellArray()
        vtk_cells.SetCells(len(triangles), vtk_triangles_array)
        self.get_polydata().SetPolys(vtk_cells)

    def get_bound(self):
        """
        Returns the bounding box of the polydata.

        The bounding box is represented as a 6-tuple:
        (xmin, xmax, ymin, ymax, zmin, zmax).

        Returns
        -------
        tuple
            A 6-tuple (xmin, xmax, ymin, ymax, zmin, zmax).
        """
        return self.polydata.GetBounds()
