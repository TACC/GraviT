/*
 * File:   readply.cpp
 * Author: jbarbosa
 *
 * Created on April 22, 2014, 10:24 AM
 */

#include <gvt/render/data/domain/reader/PlyReader.h>

#include <gvt/core/Debug.h>
#include <gvt/core/Math.h>

// Manta includes
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
// end Manta includes


#include <fstream>
#include <iostream>
#include <sstream>
#include <string>


#ifdef __USE_TAU
#include <TAU.h>
#endif

using namespace gvt::render::data::domain::reader;
using namespace gvt::render::data::primitives;

PlyReader::PlyReader(std::string filename)
{

    plyMesh = new Mesh();
    Manta::Mesh* mesh = new Manta::Mesh();
    Manta::Material *material = new Manta::Phong(Manta::Color::white() * 0.6, Manta::Color::white()* 0.8, 16, 0);
    Manta::MeshTriangle::TriangleType triangleType = Manta::MeshTriangle::KENSLER_SHIRLEY_TRI;
    //string filename(filename);
    readPlyFile(filename, Manta::AffineTransform::createIdentity(), mesh, material, triangleType);

    for(int i=0; i < mesh->vertices.size(); i++) {

                Manta::Vector v = mesh->vertices[i];
                gvt::core::math::Vector4f vertex(v[0],v[1],v[2],1.f);
                plyMesh->addVertex(vertex);
    }

    for(int i=0; i < mesh->vertex_indices.size(); i+=3) {
        plyMesh->addFace(mesh->vertex_indices[i],mesh->vertex_indices[i+1],mesh->vertex_indices[i+2]);
    }


    delete material;
    //delete mesh;
}

PlyReader::~PlyReader()
{}
