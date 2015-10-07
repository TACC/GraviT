/* 
 * File:   readply.h
 * Author: jbarbosa
 *
 * Created on April 22, 2014, 10:24 AM
 */

#ifndef GVT_RENDER_DATA_DOMAIN_READER_PLY_READER_H
#define	GVT_RENDER_DATA_DOMAIN_READER_PLY_READER_H

#include <gvt/core/Math.h>
#include <gvt/render/data/Primitives.h>

namespace gvt {
	namespace render {
		namespace data {
			namespace domain {
				namespace reader {
					/// read ply formatted geometry data
					/** read ply format files and return a Mesh object
					*/
					class PlyReader 
					{
					public:
						PlyReader(const std::string filename = "");
					    virtual ~PlyReader();

						gvt::render::data::primitives::Mesh * getMesh() { return plyMesh; }

					private:    
					    gvt::render::data::primitives::Mesh* plyMesh;
					    bool computeNormals;  
					};
				}
			}
		}
	}
}
#endif	/* GVT_RENDER_DATA_DOMAIN_READER_PLY_READER_H */

