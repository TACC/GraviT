/*
 * File:   gvt_mesh.cpp
 * Author: jbarbosa
 *
 * Created on April 20, 2014, 11:01 PM
 */
#include <gvt/render/data/Primitives.h>
#include <gvt/render/data/primitives/Mesh.h>
#ifdef __USE_TAU
#include <TAU.h>
#endif

using namespace gvt::core::math;
using namespace gvt::render::actor;
using namespace gvt::render::data;
using namespace gvt::render::data::primitives;
using namespace gvt::render::data::scene;

Mesh::Mesh(Material* mat)
  : mat(mat), haveNormals(false)
{
}

Mesh::Mesh(const Mesh& orig)
{
  mat = orig.mat;
  vertices = orig.vertices;
  normals = orig.normals;
  faces = orig.faces;
  boundingBox = orig.boundingBox;
}

Mesh::~Mesh()
{
  delete mat;
}

void Mesh::addVertexNormalTexUV(Point4f vertex, Vector4f normal, Point4f texUV)
{
#ifdef __USE_TAU
  TAU_START("Mesh::addVertexNormalTexUV");
#endif
  vertices.push_back(vertex);
  normals.push_back(normal);
  mapuv.push_back(texUV);
  boundingBox.expand(vertex);
#ifdef __USE_TAU
  TAU_STOP("Mesh::addVertexNormalTexUV");
#endif
}

void Mesh::addVertex(Point4f vertex)
{
#ifdef __USE_TAU
  TAU_START("Mesh::addVertex");
#endif
  vertices.push_back(vertex);
  boundingBox.expand(vertex);
#ifdef __USE_TAU
  TAU_STOP("Mesh::addVertex");
#endif
}

void Mesh::addNormal(Vector4f normal)
{
#ifdef __USE_TAU
  TAU_START("Mesh::addNormal");
#endif
  normals.push_back(normal);
#ifdef __USE_TAU
  TAU_STOP("Mesh::addNormal");
#endif
}

void Mesh::addTexUV(Point4f texUV)
{
#ifdef __USE_TAU
  TAU_START("Mesh::addTexUV");
#endif
  mapuv.push_back(texUV);
#ifdef __USE_TAU
  TAU_STOP("Mesh::addTexUV");
#endif
}

void Mesh::setNormal(int which, Vector4f normal)
{
#ifdef __USE_TAU
  TAU_START("Mesh::setNormal");
#endif
  GVT_ASSERT((which > vertices.size()), "Setting normal outside the bounds");
  normals[which] = normal;
#ifdef __USE_TAU
  TAU_STOP("Mesh::setNormal");
#endif
}

void Mesh::setTexUV(int which, Point4f texUV)
{
#ifdef __USE_TAU
  TAU_START("Mesh::setTexUV");
#endif
  GVT_ASSERT((which > vertices.size()), "Setting texture outside the bounds");
  this->mapuv[which] = texUV;
#ifdef __USE_TAU
  TAU_STOP("Mesh::setTexUV");
#endif
}

void Mesh::setVertex(int which, Point4f vertex, Vector4f normal, Point4f texUV)
{
#ifdef __USE_TAU
  TAU_START("Mesh::setVertex");
#endif
  GVT_ASSERT((which > vertices.size()), "Setting vertex outside the bounds");
  vertices[which] = vertex;
  boundingBox.expand(vertex);
  if(normal.length2()) normals[which] = normal;
  if(texUV.length2()) this->mapuv[which] = texUV;
#ifdef __USE_TAU
  TAU_STOP("Mesh::setVertex");
#endif
}

void Mesh::setMaterial(Material* mat)
{
#ifdef __USE_TAU
  TAU_START("Mesh::setMaterial");
#endif
  this->mat = mat;
#ifdef __USE_TAU
  TAU_STOP("Mesh::setMaterial");
#endif
}

void Mesh::addFace(int v0, int v1, int v2)
{
#ifdef __USE_TAU
  TAU_START("Mesh::addFace");
#endif
  GVT_ASSERT(v0-1 < vertices.size(), "Vertex index 0 outside bounds : " << (v0-1));
  GVT_ASSERT(v1-1 < vertices.size(), "Vertex index 1 outside bounds : " << (v1-1));
  GVT_ASSERT(v2-1 < vertices.size(), "Vertex index 2 outside bounds : " << (v2-1));
  faces.push_back(Face(v0-1,v1-1,v2-1));
#ifdef __USE_TAU
  TAU_STOP("Mesh::addFace");
#endif
}

void Mesh::addFaceToNormals(Mesh::FaceToNormals face)
{
#ifdef __USE_TAU
  TAU_START("Mesh::addFaceToNormals");
#endif
  faces_to_normals.push_back(face);
#ifdef __USE_TAU
  TAU_STOP("Mesh::addFaceToNormals");
#endif
}

void Mesh::generateNormals()
{
#ifdef __USE_TAU
  TAU_START("Mesh::generateNormals");
#endif
  if (haveNormals) {
  
#ifdef __USE_TAU
  TAU_STOP("Mesh::generateNormals");
#endif

  return;
  }
  normals.clear();
  face_normals.clear();
  faces_to_normals.clear();
  normals.resize(vertices.size());
  face_normals.resize(faces.size());
  faces_to_normals.resize(faces.size());
  for (int i = 0; i < normals.size(); ++i)
    normals[i] = Vector4f(0.0f, 0.0f, 0.0f, 0.0f);

  for (int i = 0; i < faces.size(); ++i) {
    int I = faces[i].get<0>();
    int J = faces[i].get<1>();
    int K = faces[i].get<2>();
    Vector4f a = vertices[I];
    Vector4f b = vertices[J];
    Vector4f c = vertices[K];
    Vector4f u = b - a;
    Vector4f v = c - a;
    Vector4f normal;
    normal.n[0] = u.n[1] * v.n[2] - u.n[2] * v.n[1];
    normal.n[1] = u.n[2] * v.n[0] - u.n[0] * v.n[2];
    normal.n[2] = u.n[0] * v.n[1] - u.n[1] * v.n[0];
    normal.n[3] = 0.0f;
    normal.normalize();
    face_normals[i] = normal;
    normals[I] += normal;
    normals[J] += normal;
    normals[K] += normal;
    faces_to_normals[i] = FaceToNormals(I, J, K);
  }
  for (int i = 0; i < normals.size(); ++i) normals[i].normalize();
    haveNormals = true;
#ifdef __USE_TAU
  TAU_STOP("Mesh::generateNormals");
#endif
}

Color Mesh::shadeFace(const int face_id, const Ray& r, const Vector4f& normal, const Light* lsource)
{
#ifdef __USE_TAU
  TAU_START("Color Mesh::shadeFace");
#endif
  // XXX TODO: shadeFace returns constant color, fix?

  if(!faces_to_materials.size()) {

#ifdef __USE_TAU
  TAU_STOP("Color Mesh::shadeFace");
#endif
  return shade(r,normal,lsource);
  }

  Color c(0.5f, 0.5f, 0.5f, 0.0f);
  Material* m = (faces_to_materials[face_id] ? faces_to_materials[face_id] : mat);
  if (m) {
   c = m->shade(r, normal, lsource);
  }
#ifdef __USE_TAU
  TAU_STOP("Color Mesh::shadeFace");
#endif
  return c;
}

Color Mesh::shade(const Ray& r,const Vector4f& normal,const Light* lsource)
{
  return mat->shade(r, normal, lsource);
}

Box3D Mesh::computeBoundingBox()
{
#ifdef __USE_TAU
  TAU_START("Box3D Mesh::computeBoundingBox");
#endif
  Box3D box;
  for(int i=0; i< vertices.size(); i++) {

    Point4f p = vertices[i];

    box.expand(p);
  }

  this->boundingBox = box;
#ifdef __USE_TAU
  TAU_STOP("Box3D Mesh::computeBoundingBox");
#endif

  return box;

}


