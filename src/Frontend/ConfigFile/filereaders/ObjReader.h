/* 
 * File:   ObjReader.h
 * Author: jbarbosa
 *
 * Created on January 22, 2015, 1:36 PM
 */

#ifndef OBJREADER_H
#define	OBJREADER_H

#include <vector>
#include <string>
#include <GVT/Data/primitives.h>


class ObjReader {
public:
    ObjReader(const std::string filename = "");
    ObjReader(const ObjReader& orig);
    
    
    void parse_vertex(std::string line);
    void parse_vertex_normal(std::string line);
    void parse_vertex_texture(std::string line);
    
    
    void parse_face(std::string line);
    
    virtual ~ObjReader();

//private:    
    GVT::Data::Mesh* objMesh;
    bool computeNormals;
    
    
};

#endif	/* OBJREADER_H */

