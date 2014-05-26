/* 
 * File:   readply.cpp
 * Author: jbarbosa
 * 
 * Created on April 22, 2014, 10:24 AM
 */

#include <fstream>
#include <sstream>
#include <string>

#include <iostream>

#include <GVT/common/debug.h>

#include <Model/Primitives/KenslerShirleyTriangle.h>
#include <Interface/MantaInterface.h>
#include <Interface/Scene.h>
#include <Interface/Object.h>
#include <Interface/Context.h>
#include <Core/Geometry/BBox.h>
#include <Core/Exceptions/Exception.h>
#include <Core/Exceptions/InternalError.h>
#include <Model/Groups/DynBVH.h>
#include <Model/Groups/Mesh.h>
#include <Model/Materials/Phong.h>
#include <Model/Readers/PlyReader.h>
#include <Interface/LightSet.h>
#include <Model/Lights/PointLight.h>


#include "readply.h"

GVT::Data::Mesh*  readply(std::string filename) {

    GVT::Data::Mesh* gvtmesh = new GVT::Data::Mesh();
    Manta::Mesh* mesh = new Manta::Mesh();
    Manta::Material *material = new Manta::Phong(Manta::Color::white() * 0.6, Manta::Color::white()* 0.8, 16, 0);
    Manta::MeshTriangle::TriangleType triangleType = Manta::MeshTriangle::KENSLER_SHIRLEY_TRI;
    //string filename(filename);
    readPlyFile(filename, Manta::AffineTransform::createIdentity(), mesh, material, triangleType);
    
    for(int i=0; i < mesh->vertices.size(); i++) {

                Manta::Vector v = mesh->vertices[i];
                GVT::Math::Point4f vertex(v[0],v[1],v[2],1.f);
                gvtmesh->vertices.push_back(vertex);
                gvtmesh->boundingBox.expand(vertex);
    }
    
    for(int i=0; i < mesh->vertex_indices.size(); i+=3) {
        gvtmesh->faces.push_back(GVT::Data::Mesh::face(mesh->vertex_indices[i],mesh->vertex_indices[i+1],mesh->vertex_indices[i+2]));
    }
    
    
    delete material;
    //delete mesh;

    
    return gvtmesh;

}