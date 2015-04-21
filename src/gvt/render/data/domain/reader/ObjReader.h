/* 
 * File:   ObjReader.h
 * Author: jbarbosa
 *
 * Created on January 22, 2015, 1:36 PM
 */

#ifndef GVT_RENDER_DATA_DOMAIN_READER_OBJ_READER_H
#define	GVT_RENDER_DATA_DOMAIN_READER_OBJ_READER_H

#include <gvt/render/data/Primitives.h>

#include <string>
#include <vector>

namespace gvt {
	namespace render {
		namespace data {
			namespace domain {
				namespace reader {
					class ObjReader {
					public:
					    ObjReader(const std::string filename = "");
					    virtual ~ObjReader();

						gvt::render::data::primitives::Mesh * getMesh() { return objMesh; }

					private:    
					    void parseVertex(std::string line);
					    void parseVertexNormal(std::string line);
					    void parseVertexTexture(std::string line);
					    
					    
					    void parseFace(std::string line);
					    
					    gvt::render::data::primitives::Mesh* objMesh;
					    bool computeNormals;		    
				    };
				}
			}
		}
	}
}

#endif	/* GVT_RENDER_DATA_DOMAIN_READER_OBJ_READER_H */

