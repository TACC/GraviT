//
//  Dataset.C
//

#include "Dataset.h"

#include <GVT/Data/primitives.h>
#include <GVT/Data/scene/Utils.h>
#include <GVT/Domain/domains.h>

//#include "VolumeDomain.h" // XXX TODO remove when this made a subclass

#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include <GVT/common/debug.h>
namespace GVT {
    namespace Dataset {

        template<>
        bool Dataset<GVT::Domain::GeometryDomain>::init() {
            float min[3];
            float max[3];
            int size[3];
            vector< vector<int> > sizes;
            vector< vector<int> > offsets;
            GVT_ASSERT(conf_filename.size() > 0, "ERROR: Configuration file for dataset not set");

            fstream conf;
            conf.open(conf_filename.c_str(), fstream::in);
            GVT_ASSERT(conf.is_open(), "ERROR: Could not open file '" << conf_filename << "'");
            size_t ptr = conf_filename.find_last_of("/\\");
            string conf_dir = conf_filename.substr(0, ptr);

            GVT_DEBUG(DBG_ALWAYS, "opened file '" << conf_filename << "' with dir '" << conf_dir << "'");

            // read chunk filenames and layout data
            while (conf.good()) {
                string line;
                getline(conf, line);

                GVT_DEBUG(DBG_ALWAYS, "got line: '" << line << "'");

                if (line.size() > 0) {

                    stringstream buf;
                    string file = conf_dir;
                    size_t pos, lastpos;
                    pos = line.find_first_of(" ");
                    file += "/";
                    file += line.substr(0, pos);
                    this->files.push_back(file);

                    // next int is the var number
                    lastpos = pos + 1;
                    pos = line.find_first_of(" ", lastpos);
                    if (pos == string::npos) {
                        cerr << "ERROR: didn't find var number in chunk line: '" << line << "'" << endl;
                        return false;
                    }
                    buf.str(line.substr(lastpos, pos));
                    int var_num;
                    buf >> var_num;
                    GVT_DEBUG(DBG_ALWAYS, "    var_num: " << var_num);

                    // next three ints are the chunk dimensions
                    vector<int> dims;
                    GVT_DEBUG(DBG_ALWAYS, "    dims: ");
                    for (int i = 0; i < 3; ++i) {
                        lastpos = pos + 1;
                        pos = line.find_first_of(" ", lastpos);
                        if (pos == string::npos) {
                            cerr << "ERROR: didn't find dim " << i << " in chunk line: '" << line << "'" << endl;
                            return false;
                        }
                        buf.str(line.substr(lastpos, pos));
                        int d;
                        buf >> d;
                        dims.push_back(d);
                        GVT_DEBUG(DBG_ALWAYS, d << " ");
                    }

                    sizes.push_back(dims);

                    // next three ints are the chunk position offset
                    vector<int> offs;
                    GVT_DEBUG(DBG_ALWAYS, "    offs: ");
                    for (int i = 0; i < 2; ++i) {
                        lastpos = pos + 1;
                        pos = line.find_first_of(" ", lastpos);
                        if (pos == string::npos) {
                            cerr << "ERROR: didn't find dim " << i << " in chunk line: '" << line << "'" << endl;
                            return false;
                        }
                        buf.str(line.substr(lastpos, pos));
                        int d;
                        buf >> d;
                        offs.push_back(d);
                        if (d < min[i]) min[i] = d;
                        if ((d + dims[i]) > max[i]) max[i] = d + dims[i];
                        if ((d + dims[i]) > size[i]) size[i] = d + dims[i];
                        GVT_DEBUG(DBG_ALWAYS, d << " ");
                    }
                    // special case for last int on line
                    {
                        buf.str(line.substr(pos + 1));
                        int d;
                        buf >> d;
                        offs.push_back(d);
                        if (d < min[2]) min[2] = d;
                        if ((d + dims[2]) > max[2]) max[2] = d + dims[2];
                        if ((d + dims[2]) > size[2]) size[2] = d + dims[2];
                        GVT_DEBUG(DBG_ALWAYS, d << " ");
                    }

                    offsets.push_back(offs);

                }
            }

            return true;
        }


//        template<>
//        GVT::Domain::Domain*
//        Dataset<GVT::Domain::GeometryDomain>::getDomain(int id) {
//            if (id < 0 || id >= files.size()) {
//                cerr << "ERROR: invalid domain id '" << id << "' passed to GetDomain" << endl;
//                return NULL;
//            }
//
//            // check domain cache first
//            // XXX TODO remove VolumeDomain, make this a subclass
//            map< int, GVT::Domain::GeometryDomain >::iterator it = dom_cache.find(id);
//            if (it != dom_cache.end())
//                return &(it->second);
//
//            // if domain isn't in cache, create it
//            vector<float> min, max;
//            for (int i = 0; i < 3; ++i) {
//                min.push_back((float) offsets[id][i]);
//                max.push_back((float) (offsets[id][i] + sizes[id][i]));
//            }
//            dom_cache[id] = GVT::Domain::GeometryDomain(); // XXX TODO fix when this made a subclass
//
//            //cout << dom_cache[id] << endl;
//
//            return &(dom_cache[id]);
//        }
    }
}