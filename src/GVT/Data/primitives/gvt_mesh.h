/* 
 * File:   gvt_mesh.h
 * Author: jbarbosa
 *
 * Created on April 20, 2014, 11:01 PM
 */

#ifndef GVT_MESH_H
#define	GVT_MESH_H


#include <GVT/Data/primitives/gvt_material.h>

#include <vector>

#include <boost/tuple/tuple.hpp>
#include <boost/container/vector.hpp>

#include "gvt_bbox.h"

namespace GVT {
    namespace Data {

        
        class AbstractMesh {
            
        public:
            AbstractMesh() {
                
            }
            AbstractMesh(const AbstractMesh&) {
                
            }
            ~AbstractMesh() {
                
            }
            
            
            virtual AbstractMesh* getMesh() {
                return this;
            }
            
            virtual GVT::Data::box3D* getBoundingBox() {
                return NULL;
            }
            
        };
        
        class Mesh : public AbstractMesh {
        public:

            typedef boost::tuple<int, int, int> face;

            Mesh(GVT::Data::Material* mat=NULL);
            Mesh(const Mesh& orig);


            virtual ~Mesh();
            virtual void addVertex(GVT::Math::Point4f vertex, GVT::Math::Vector4f normal = GVT::Math::Vector4f(), GVT::Math::Point4f texUV = GVT::Math::Point4f());
            virtual void pushVertex(int which, GVT::Math::Point4f vertex, GVT::Math::Vector4f normal = GVT::Math::Vector4f(), GVT::Math::Point4f texUV = GVT::Math::Point4f());
            virtual void pushNormal(int which, GVT::Math::Vector4f normal = GVT::Math::Point4f());
            virtual void pushTexUV(int which, GVT::Math::Point4f texUV = GVT::Math::Point4f());
            virtual void setMaterial(GVT::Data::Material* mat);
            virtual void addFace(int v0, int v1, int v2);
            
            virtual GVT::Data::Color shade(GVT::Data::ray&  r, GVT::Math::Vector4f normal, GVT::Data::LightSource* lsource);
            GVT::Data::Material* mat;

            boost::container::vector<GVT::Math::Vector4f> vertices;
            boost::container::vector<GVT::Math::Vector4f> mapuv;
            boost::container::vector<GVT::Math::Vector4f> normals;
            boost::container::vector<GVT::Data::Mesh::face> faces;

            GVT::Data::box3D boundingBox;

        };
    }
}



#endif	/* GVT_MESH_H */

