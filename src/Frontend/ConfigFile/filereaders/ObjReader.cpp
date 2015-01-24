/* 
 * File:   ObjReader.cpp
 * Author: jbarbosa
 * 
 * Created on January 22, 2015, 1:36 PM
 */

#include "ObjReader.h"
#include <iostream>
#include <fstream>
#include <sstream>

std::vector<std::string> split(const std::string &s, char delim, std::vector<std::string> &elems) {
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        if (item.length() > 0) {
            elems.push_back(item);
        }
    }
    return elems;
}

ObjReader::ObjReader(const std::string filename) : computeNormals(false) {

    GVT_ASSERT(filename.size() > 0, "Invalid filename");
    std::fstream file;
    file.open(filename.c_str());
    GVT_ASSERT(file.good(), "Error loading obj file " << filename);

    objMesh = new GVT::Data::Mesh(new GVT::Data::Lambert(GVT::Math::Vector4f(0.5,0.5,0.5,1.0)));
    
    //objMesh->setMaterial(new GVT::Data::Lambert());
    
    while (file.good()) {
        std::string line;
        std::getline(file, line);


        if (line.find("#") == 0) continue;

        if (line.find("v") == 0) {
            parse_vertex(line);
            continue;
        }
        if (line.find("vn") == 0) {
            parse_vertex_normal(line);
            continue;
        }
        if (line.find("vt") == 0) {
            parse_vertex_texture(line);
            continue;
        }
        if (line.find("f") == 0) {
            parse_face(line);
            continue;
        }
    }

    if(computeNormals) objMesh->generateNormals();
    objMesh->computeBoundingBox();





}

void ObjReader::parse_vertex(std::string line) {
    std::vector<std::string> elems;
    split(line, ' ', elems);
    GVT_ASSERT(elems.size() == 4, "Error parsing vertice");
    objMesh->vertices.push_back(GVT::Math::Vector4f(std::atof(elems[1].c_str()), std::atof(elems[2].c_str()), std::atof(elems[3].c_str()), 1.0f));
}

void ObjReader::parse_vertex_normal(std::string line) {
    std::vector<std::string> elems;
    split(line, ' ', elems);
    GVT_ASSERT(elems.size() == 4, "Error parsing vertice normal");
    objMesh->normals.push_back(GVT::Math::Vector4f(std::atof(elems[1].c_str()), std::atof(elems[2].c_str()), std::atof(elems[3].c_str()), 1.0f));
}

void ObjReader::parse_vertex_texture(std::string line) {
    std::vector<std::string> elems;
    split(line, ' ', elems);
    GVT_ASSERT(elems.size() == 3, "Error parsing texture map");
    objMesh->mapuv.push_back(GVT::Math::Vector4f(std::atof(elems[1].c_str()), std::atof(elems[2].c_str()), 0, 0));
}

void ObjReader::parse_face(std::string line) {
    std::vector<std::string> elems;
    split(line, ' ', elems);
    GVT_ASSERT(elems.size() == 4, "Error parsing face");
    
    int v1,n1,t1;
    int v2,n2,t2;
    int v3,n3,t3;

    v1 = n1 = t1 = 0;
    v2 = n2 = t2 = 0;
    v3 = n3 = t3 = 0;
    
    /* can be one of %d, %d//%d, %d/%d, %d/%d/%d %d//%d */
    if (std::strstr(elems[1].c_str(), "//")) {
        std::sscanf(elems[1].c_str(), "%d//%d", &v1, &n1);
        std::sscanf(elems[2].c_str(), "%d//%d", &v2, &n2);
        std::sscanf(elems[3].c_str(), "%d//%d", &v3, &n3);
        objMesh->faces_to_normals.push_back(GVT::Data::Mesh::face_to_normals(n1-1,n2-1,n3-1));
    } else if (std::sscanf(elems[1].c_str(), "%d/%d/%d", &v1, &t1, &n1) == 3) {
        /* v/t/n */
        std::sscanf(elems[2].c_str(), "%d/%d/%d", &v2, &t2, &n2);
        std::sscanf(elems[3].c_str(), "%d/%d/%d", &v3, &t3, &n3);
        objMesh->addFace(v1,v2,v3);
        objMesh->faces_to_normals.push_back(GVT::Data::Mesh::face_to_normals(n1-1,n2-1,n3-1));
//        objMesh->faces_uv.push_back(GVT::Data::Mesh::face_uv(t1-1,t2-1,t3-1));
    } else if (std::sscanf(elems[1].c_str(), "%d/%d", &v1, &t1) == 2) {
        /* v/t */
        std::sscanf(elems[2].c_str(), "%d/%d", &v2, &t2);
        std::sscanf(elems[3].c_str(), "%d/%d", &v3, &t3);
        objMesh->addFace(v1,v2,v3);
//        objMesh->faces_uv.push_back(GVT::Data::Mesh::face_uv(t1-1,t2-1,t3-1));
        computeNormals = true;
    } else {
        /* v */
        std::sscanf(elems[1].c_str(), "%d", &v1);
        std::sscanf(elems[2].c_str(), "%d", &v2);
        std::sscanf(elems[3].c_str(), "%d", &v3);        
        computeNormals = true;
    }
    
    objMesh->addFace(v1,v2,v3);
    
   
}

ObjReader::ObjReader(const ObjReader& orig) {
}

ObjReader::~ObjReader() {
}

