import numpy as np
import sys
import os
import vtk
from vtk import *

# load mesh
# filename = sys.argv[1]

class Reader:
  def __init__(self):
    self.polydata = None
    self.reader = None
    self.vertices = None
    self.triangles = None
    self.bounds = None
    self.bounds_min = None
    self.bounds_max = None
    self.center = None
    self.diagonal = None
    self.mesh_name = None

  def print_stats(self):
    print('mesh name:' + self.mesh_name)
    print('bounds: min(' + str(self.bounds_min) + ') max(' + str(self.bounds_max) + ')')
    print('center: ' + str(self.center))
    print('number of vertices: ' + str(len(self.vertices) / 3))
    print('number of triangles: ' + str(len(self.triangles) / 3))

  def read(self, filename):
    if os.path.isfile(filename) == False:
      print(filename + ' is not a file.')
      raise ValueError('not a file.')

    self.mesh_name = os.path.basename(filename).split('.')[0]
    mesh_type = os.path.basename(filename).split('.')[1]
   
    if mesh_type in ['ply']: 
      self.reader = vtk.vtkPLYReader()

    elif mesh_type in ['obj']: 
      self.reader = vtk.vtkOBJReader()

    elif mesh_type in ['vtp']: 
      self.reader = vtk.vtkXMLPolyDataReader()

    else:
      print(mesh_type + 'is unsupported.')
      raise ValueError('unsupported file extension.')
    
    self.reader.SetFileName(filename)
    self.reader.Update()

    # print('filename', filename)
    
    self.polydata = self.reader.GetOutput()
    
    num_points = self.polydata.GetNumberOfPoints()
    # print('num_points', num_points)
    
    # TODO: enable color
    # point_data = self.polydata.GetPointData()
    # rgb = point_data.GetArray('RGB')
    # rgb.GetTuple(0)

    bounds = self.polydata.GetBounds()
    self.bounds = bounds
    self.bounds_min = np.array([bounds[0], bounds[2], bounds[4]])
    self.bounds_max = np.array([bounds[1], bounds[3], bounds[5]])

    self.center = np.array(self.polydata.GetCenter())
    self.diagonal = self.polydata.GetLength()
    
    points = self.polydata.GetPoints()
   
    # vertices 
    self.vertices = np.zeros(self.polydata.GetNumberOfPoints() * 3 , dtype = np.float32)
    
    for i in range(self.polydata.GetNumberOfPoints()):
      pos = points.GetPoint(i)
      for j in range(len(pos)):
        self.vertices[i * 3 + j] = pos[j]
    
    # triangles
    self.triangles = np.zeros(self.polydata.GetNumberOfCells() * 3 , dtype = np.uint32)
    
    for i in range(self.polydata.GetNumberOfCells()):
      c = self.polydata.GetCell(i)
      if (c.GetNumberOfPoints() != 3):
        print("Number of vertices per cell: ", c.GetNumberOfPoints())
        raise ValueError('The number of vertices is not 3.')
      for j in range(c.GetNumberOfPoints()):
        self.triangles[i * 3 + j] = c.GetPointId(j)
   
    # gvt indices starting from 1 
    for x in np.nditer(self.triangles, op_flags=['readwrite']):
      x[...] = x + 1
    
