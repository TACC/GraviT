cimport cython


import sys
import ctypes
import numpy as np
cimport numpy as np

from libcpp.string cimport string
from libc.stdlib cimport malloc, free
from libcpp cimport bool


cdef extern from "api.h":
  void _gvtInit"gvtInit"(int argc, char** argv)
  void _readPly"readPly"(string dirname, bool dist, float* world_bounds)
  void _createMesh"createMesh"(string)
  void _addMeshVertices"addMeshVertices"(string name, unsigned &n, float *vertices)
  void _addMeshTriangles"addMeshTriangles"( string name,  unsigned &n,  unsigned *triangles)
  void _addMeshFaceNormals"addMeshFaceNormals"( string name,  unsigned &n,  float *normals)
  void _addMeshVertexNormals"addMeshVertexNormals"( string name,  unsigned &n,  float *normals)
  void _finishMesh"finishMesh"( string name, bool compute_normal)
  void _addMeshMaterial"addMeshMaterial"( string name,  unsigned mattype,  float *kd,  float alpha)
  void _addMeshMaterial2"addMeshMaterial"( string name,  unsigned mattype,  float *kd,  float *ks,
                        float alpha )
  void _addMeshMaterials"addMeshMaterials"( string name,  unsigned n,  unsigned *mattype,  float *kd,
                         float *ks,  float *alpha)
  void _addInstance"addInstance"(string name,  float *m)
  void _addPointLight"addPointLight"(string name,  float *pos,  float *color)
  void _addAreaLight"addAreaLight"(string name,  float *pos,  float *color,  float *n, float w, float h)
  void _modifyLight"modifyLight"(string name,  float *pos,  float *color)
  void _modifyLight"modifyLight"(string name,  float *pos,  float *color,  float *n, float w, float h)
  void _addCamera"addCamera"(string name,  float *pos,  float *focus,  float *up, float fov, int depth,
                 int samples, float jitter)
  void _modifyCamera"modifyCamera"(string name,  float *pos,  float *focus,  float *up, float fov, int depth,
                    int samples, float jitter)
  void _modifyCamera"modifyCamera"(string name,  float *pos,  float *focus,  float *up, float fov)
  void _addFilm"addFilm"(string name, int w, int h, string path)
  void _modifyFilm"modifyFilm"(string name, int w, int h, string path)
  void _render"render"(string name)
  void _writeimage"writeimage"(string name, string output)
  void _addRenderer"addRenderer"(string name, int adapter, int schedule)
#void gvtInit(int &argc, char **&argv)

def gvtInit():
  # cdef char **c_argv
  # c_argv = <char**>malloc(sizeof(char*) * len(sys.argv))
  # for idx, s in enumerate(sys.argv):
  #   c_argv[idx] = PyUnicode_AsEncodedString(s)
  # free(c_argv)

  _gvtInit(0,NULL);


  # LP_c_char = ctypes.POINTER(ctypes.c_char)
  # LP_LP_c_char = ctypes.POINTER(LP_c_char)
  #
  # _gvtInit.argtypes = (ctypes.c_int, # argc
  #                           LP_LP_c_char) # argv)
  # argc = len(sys.argv)
  # argv = (LP_c_char * (argc + 1))()
  # for i, arg in enumerate(sys.argv):
  #   enc_arg = arg.encode('utf-8')
  #   argv[i] = ctypes.create_string_buffer(enc_arg)
  # _gvtInit(argc,argv)

def readPly(str dirname, bool dist, np.ndarray[float, ndim=1, mode="c"] world_bounds):
  _readPly(dirname.encode(), dist, <float*> world_bounds.data)

def createMesh(str name):
  _createMesh(name.encode())

def addMeshVertices(str name, int size, np.ndarray[float, ndim=1, mode="c"] vertices not None):
  _addMeshVertices(name.encode(),size,<float*> vertices.data)

def addMeshTriangles(str name, unsigned n,  np.ndarray[unsigned, ndim=1, mode="c"] triangles):
  _addMeshTriangles(name.encode(),  n,  <unsigned *> triangles.data)

def addMeshFaceNormals( str name, unsigned n,   np.ndarray[float, ndim=1, mode="c"] normals):
  _addMeshFaceNormals(name.encode(), n, <float*> normals.data)

def addMeshVertexNormals( str name,  unsigned n,  np.ndarray[float, ndim=1, mode="c"] normals):
  _addMeshVertexNormals(name.encode(),n, <float*> normals.data)

def finishMesh( str name, bool compute_normal = True):
  _finishMesh(name.encode(),compute_normal)

def addMeshMaterialLambert( str name,  unsigned mattype,  np.ndarray[float, ndim=1, mode="c"] kd,  float alpha):
  _addMeshMaterial(name.encode() ,mattype, <float*> kd.data, alpha)

def addMeshMaterialSpecular( str name,  unsigned mattype,  np.ndarray[float, ndim=1, mode="c"] kd,  np.ndarray[float, ndim=1, mode="c"] ks, float alpha ):
  _addMeshMaterial2(name.encode(), mattype, <float*> kd.data, <float*> ks.data, alpha)

# def addMeshMaterials( str name,  unsigned n,  np.ndarray[uint32, ndim=1, mode="c"] mattype,  np.ndarray[float, ndim=1, mode="c"] kd, np.ndarray[float, ndim=1, mode="c"] ks,  np.ndarray[float, ndim=1, mode="c"] alpha):
#   _addMeshMaterials(name.encode(),n,<unsigned*> mattype.data, <float*> kd.data, <float*> ks.float, <float*> alpha.data)

def addInstance(str name,  np.ndarray[float, ndim=1, mode="c"] m):
  _addInstance(name.encode(),<float*> m.data)

def addPointLight(str name,  np.ndarray[float, ndim=1, mode="c"] pos,  np.ndarray[float, ndim=1, mode="c"] color):
  _addPointLight(name.encode(),<float*> pos.data, <float*> color.data)

def addAreaLight(str name,  np.ndarray[float, ndim=1, mode="c"] pos,  np.ndarray[float, ndim=1, mode="c"] color,  np.ndarray[float, ndim=1, mode="c"] n, float w, float h):
  _addAreaLight(name.encode(),<float*> pos.data, <float*> color.data, <float*> n.data, w, h)

def modifyLight(str name,  np.ndarray[float, ndim=1, mode="c"] pos,  np.ndarray[float, ndim=1, mode="c"] color):
  _modifyLight(name.encode(),<float*>pos.data, <float*> color.data)

def modifyLight(str name,  np.ndarray[float, ndim=1, mode="c"] pos,  np.ndarray[float, ndim=1, mode="c"] color,  np.ndarray[float, ndim=1, mode="c"] n, float w, float h):
  _modifyLight(name.encode(),<float*>pos.data,<float*>color.data,<float*>n.data,w,h)

def addCamera(str name,  np.ndarray[float, ndim=1, mode="c"]  pos,  np.ndarray[float, ndim=1, mode="c"]  focus,  np.ndarray[float, ndim=1, mode="c"]  up, float fov, int depth, int samples, float jitter):
  _addCamera(name.encode(), <float*> pos.data, <float*>focus.data,<float*>up.data,fov,depth,samples,jitter)

def modifyCamera(str name,  np.ndarray[float, ndim=1, mode="c"] pos,  np.ndarray[float, ndim=1, mode="c"]  focus,  np.ndarray[float, ndim=1, mode="c"]  up, float fov):
  _modifyCamera(name.encode(), <float*> pos.data, <float*>focus.data,<float*>up.data,fov)

def addFilm(str name, int w, int h, str path):
  _addFilm(name.encode(),w,h,path.encode())

def modifyFilm(str name, int w, int h, str path):
  _modifyFilm(name.encode(),w,h,path.encode())

def addRenderer(str name, int adapter, int schedule):
  _addRenderer(name.encode(),adapter,schedule)

def render(str name):
  _render(name.encode())

def writeimage(str name, str output):
  _writeimage(name.encode(),output.encode())
