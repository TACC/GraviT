/* 
 * File:   gvt_mesh.cpp
 * Author: jbarbosa
 * 
 * Created on April 20, 2014, 11:01 PM
 */
#include <GVT/Data/primitives.h>
#include "gvt_mesh.h"
namespace GVT {
    namespace Data {
        
        Mesh::Mesh(GVT::Data::Material* mat) : mat(mat) {
        }

        Mesh::Mesh(const Mesh& orig) {
            mat = orig.mat;
            vertices = orig.vertices;
            normals = orig.normals;
            faces = orig.faces;
            boundingBox = orig.boundingBox;
        }

        Mesh::~Mesh() {
        }
        
        void Mesh::addVertex(GVT::Math::Point4f vertex, GVT::Math::Vector4f normal, GVT::Math::Point4f texUV) {
            vertices.push_back(vertex);
            normals.push_back(normal);
            mapuv.push_back(texUV);
            boundingBox.expand(vertex);
        }
        
        void Mesh::pushNormal(int which, GVT::Math::Vector4f normal) {
            GVT_ASSERT((which > vertices.size()), "Adding normal outside the bounds");
            normals[which] = normal;
        }
        
        void Mesh::pushTexUV(int which, GVT::Math::Point4f texUV) {
            GVT_ASSERT((which > vertices.size()), "Adding texture outside the bounds");
            this->mapuv[which] = texUV;
        }
        
        void Mesh::pushVertex(int which, GVT::Math::Point4f vertex, GVT::Math::Vector4f normal, GVT::Math::Point4f texUV) {
            GVT_ASSERT((which > vertices.size()), "Adding vertex outside the bounds");
            vertices[which] = vertex;
            boundingBox.expand(vertex);
            if(normal.length2()) normals[which] = normal;
            if(texUV.length2()) this->mapuv[which] = texUV;
        }
        
        void Mesh::setMaterial(GVT::Data::Material* mat) {
            this->mat = mat; 
        }
        
        void Mesh::addFace(int v0, int v1, int v2) {
            GVT_ASSERT(v0 < vertices.size(), "Vertex index 0 outside bounds");
            GVT_ASSERT(v1 < vertices.size(), "Vertex index 1 outside bounds");
            GVT_ASSERT(v2 < vertices.size(), "Vertex index 2 outside bounds");
            
            
            
            faces.push_back(face(v0,v1,v2));
        }
 
        GVT::Data::Color Mesh::shade(GVT::Data::ray&  r, GVT::Math::Vector4f normal, GVT::Data::LightSource* lsource) {
            return mat->shade(r,normal,lsource);
        }
        
        
    }
}


