//
//  Dataset.C
//

#include <Domain/MantaDomain.h>
#include <GVT/DataSet/Dataset.h>
#include <GVT/Data/primitives.h>
#include <GVT/Data/scene/Utils.h>
#include <GVT/Domain/domains.h>

#include <GVT/common/utils.h>
#include <GVT/Data/primitives.h>

//#include "VolumeDomain.h" // XXX TODO remove when this made a subclass

#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

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

#include <GVT/Data/primitives.h>
#include <GVT/DataSet/Dataset.h>
#include <Frontend/ConfigFile/Dataset/Dataset.h>
#include <gvtmanta.h>

namespace GVT {
    namespace Dataset {

        template<>
        bool Dataset<GVT::Domain::MantaDomain>::init() {
            float min[3];
            float max[3];
            int size[3];
            vector< vector<int> > sizes;
            vector< vector<int> > offsets;
            std::cout << "Opened file" << std::endl;
            GVT_ASSERT(conf_filename.size() > 0, "configuration file for dataset not set : " << conf_filename );
            
            fstream conf;
            conf.open(conf_filename.c_str(), fstream::in);
            GVT_ASSERT(conf.is_open(), "ERROR: Could not open file '" << conf_filename << "'");;
            // pull absolute path from conf_filename to prepend to chunk filenames
            size_t ptr = conf_filename.find_last_of("/\\");
            string conf_dir = conf_filename.substr(0, ptr);

            GVT_DEBUG(DBG_LOW, "opened file '" << conf_filename << "' with dir '" << conf_dir << "'");

            // read chunk filenames and layout data
            while (conf.good()) {
                string line;
                getline(conf, line);
                if (line.size() > 0 && line.substr(0, 1) != "#") {
                    GVT_DEBUG(DBG_LOW, "got line: '" << line << "'");
                    line+=" ";
                    stringstream buf;
                    string file = conf_dir;
                    size_t pos, lastpos;
                    pos = line.find_first_of(" ");
                    file += "/";
                    file += line.substr(0, pos);
                    this->files.push_back(file);


                    GVT::Math::Vector3f t;
                    int var_num;
                    for (int i = 0; i < 3; ++i) {
                        lastpos = pos + 1;
                        pos = line.find_first_of(" ", lastpos);
                        GVT_ASSERT(pos != string::npos, "ERROR: didn't find translate " << i << " in chunk line: '" << line << "'");
                        buf.str(line.substr(lastpos, pos));
                        buf >> t[i];
                    }

                    GVT::Math::Vector4f r(0.f, 0.f, 0.f, 0.f);
                    for (int i = 0; i < 4; ++i) {
                        lastpos = pos + 1;
                        pos = line.find_first_of(" ", lastpos);
                        GVT_ASSERT(pos != string::npos, "ERROR: didn't find rotate " << i << " in chunk line: '" << line << "' (if i = 3, probably needs a extra space at the end of the line)");
                        buf.str(line.substr(lastpos, pos));
                        buf >> r[i];
                    }

                    GVT::Math::AffineTransformMatrix<float> m = GVT::Math::AffineTransformMatrix<float>::createTranslation(t[0], t[1], t[2]);
 
                    GVT::Domain::MantaDomain* dom = new GVT::Domain::MantaDomain(file, m);
                    GVT_DEBUG(DBG_LOW,"Domain bounding box : " << dom->getWorldBoundingBox());
                    this->addDomain(dom);
                } else {
                    GVT_DEBUG(DBG_LOW,"Ignored line : '" << line << "'");
                }
            } 
            return true;
        }

//        template<>
//        GVT::Domain::Domain*
//        Dataset<GVT::Domain::MantaDomain>::getDomain(int id) {
//            if (id < 0 || id >= files.size()) {
//                cerr << "ERROR: invalid domain id '" << id << "' passed to GetDomain" << endl;
//                return NULL;
//            }
//            cout << "Get domain " << files[id] << endl;
//            map< int, GVT::Domain::MantaDomain >::iterator it = dom_cache.find(id);
//            if (it != dom_cache.end()) {
//                cout << "In cache " << files[id] << endl;
//                return &(it->second);
//            }
//            dom_cache[id] = GVT::Domain::MantaDomain(files[id], dom_model[id]);
//            cout << "Loaded domain " << files[id] << endl;
//            return &(dom_cache[id]);
//        }

    }
}

