cimport cython


import sys
import ctypes
import numpy as np
cimport numpy as np

from libcpp.string cimport string
from libc.stdlib cimport malloc, free
from libcpp cimport bool


cdef extern from "gravit/api.h" namespace "api":
  void _gvtInit "api::gvtInit"(int argc, char** argv)
  void _createMesh "api::createMesh"(string)
  void _addMeshVertices "api::addMeshVertices"(string name, unsigned &n, float *vertices)
  void _addMeshTriangles "api::addMeshTriangles"( string name,  unsigned &n,  unsigned *triangles)
  void _addMeshFaceNormals "api::addMeshFaceNormals"( string name,  unsigned &n,  float *normals)
  void _addMeshVertexNormals "api::addMeshVertexNormals"( string name,  unsigned &n,  float *normals)
  void _finishMesh "api::finishMesh"( string name, bool compute_normal)
  void _addMeshMaterial "api::addMeshMaterial"( string name,  unsigned mattype,  float *kd,  float alpha)
  void _addMeshMaterial2 "api::addMeshMaterial"( string name,  unsigned mattype,  float *kd,  float *ks, float alpha )
  void _addMeshMaterials "api::addMeshMaterials"( string name,  unsigned n,  unsigned *mattype,  float *kd, float *ks,  float *alpha)
  void _addInstance "api::addInstance"(string name, string meshname, float *m)
  void _createVolume"api::createVolume"(string name)
  void _addVolumeTransferFunctions"api::addVolumeTransferFunctions"(string name, string colortfname, string opacityfname, float low, float high)
  void _addVolumeSamples"api::addVolumeSamples"(string name, float *samples, int *counts, float *origin, float* deltas, float samplingrate, double *bounds)
  void _addAmrSubgrid"api::addAmrSubgrid"(string name, int gridid, int level, float *samples, int *counts, float *origin, float *deltas)
  void _addPointLight "api::addPointLight"(string name,  float *pos,  float *color)
  void _addAreaLight "api::addAreaLight"(string name,  float *pos,  float *color,  float *n, float w, float h)
  void _modifyLight "api::modifyLight"(string name,  float *pos,  float *color)
  void _modifyLight "api::modifyLight"(string name,  float *pos,  float *color,  float *n, float w, float h)
  void _addCamera "api::addCamera"(string name,  float *pos,  float *focus,  float *up, float fov, int depth, int samples, float jitter)
  void _modifyCamera "api::modifyCamera"(string name,  float *pos,  float *focus,  float *up, float fov, int depth, int samples, float jitter)
  void _modifyCamera "api::modifyCamera"(string name,  float *pos,  float *focus,  float *up, float fov)
  void _addFilm "api::addFilm"(string name, int w, int h, string path)
  void _modifyFilm "api::modifyFilm"(string name, int w, int h, string path)
  void _render "api::render"(string name)
  void _writeimage "api::writeimage"(string name, string output)
  void _addRenderer "api::addRenderer"(string name, int adapter, int schedule,
          string camera_name, string film_name, bool volume)
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

def createMesh(str name):
  _createMesh(name.encode())
def addVolumeTransferFunctions(str name, str colortfname, str opacityfname, float low, float high):
   _addVolumeTransferFunctions(name.encode(),colortfname.encode(),opacityfname.encode(),low,high)
def addVolumeSamples(str name, np.ndarray[float, ndim=1,mode="c"] samples, np.ndarray[int,ndim=1,mode="c"] counts, np.ndarray[float,ndim=1,mode="c"] origin, np.ndarray[float,ndim=1,mode="c"] deltas,float samplingrate,np.ndarray[double,ndim=1,mode="c"] bounds):
   _addVolumeSamples(name.encode(),<float*>samples.data,<int*>counts.data,<float*>origin.data,<float*>deltas.data,samplingrate,<double*>bounds.data)
def addAmrSubgrid(str name, int gridid, int level,np.ndarray[float,ndim=1,mode="c"] samples, np.ndarray[int,ndim=1,mode="c"] counts, np.ndarray[float,ndim=1,mode="c"] origin,np.ndarray[float,ndim=1,mode="c"] deltas]):
    _addAmrSubgrid(name.encode(),gridid,level,<float*>samples.data,<int*>counts.data,<float*>origin.data,<float*>deltas.data)
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

def addInstance(str s, str name,  np.ndarray[float, ndim=1, mode="c"] m):
  _addInstance(s.encode(),name.encode(),<float*> m.data)
def createVolume(str name):
      _createVolume(name.encode())

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

def addRenderer(str name, int adapter, int schedule, str camera_name,
        str film_name, bool volume):
  _addRenderer(name.encode(),adapter,schedule,camera_name.encode(),
          film_name.encode(),volume)

def render(str name):
  _render(name.encode())

def writeimage(str name, str output):
  _writeimage(name.encode(),output.encode())
