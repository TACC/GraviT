//
//  Dataset.C
//

#include "Dataset.h"

#include <GVT/Data/primitives/gvt_bbox.h>
#include <GVT/Data/primitives.h>
#include <GVT/Data/scene/Utils.h>
#include <GVT/Domain/domains.h>

//#include "GVT::Domain::VolumeDomain.h" // XXX TODO remove when this made a subclass

#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include <GVT/common/debug.h>

namespace GVT {
    namespace Dataset {

        template<>
        bool Dataset<GVT::Domain::VolumeDomain>::Init() {
            if (conf_filename.size() == 0) {
                GVT_DEBUG(DBG_ALWAYS,"ERROR: Configuration file for dataset not set");
                return false;
            }

            fstream conf;

            conf.open(conf_filename.c_str(), fstream::in);
            if (!conf.is_open()) {
                GVT_DEBUG(DBG_ALWAYS,"ERROR: Could not open file '" << conf_filename << "'");
                return false;
            }

            // pull absolute path from conf_filename to prepend to chunk filenames
            size_t ptr = conf_filename.find_last_of("/\\");
            string conf_dir = conf_filename.substr(0, ptr);

            GVT_DEBUG(DBG_LOW,"opened file '" << conf_filename << "' with dir '" << conf_dir << "'");

            int domit = 0;
            // read chunk filenames and layout data
            while (conf.good()) {
                string line;
                getline(conf, line);

                GVT_DEBUG(DBG_LOW,"got line: '" << line << "'");

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
                        GVT_DEBUG(DBG_LOW,"ERROR: didn't find var number in chunk line: '" << line << "'");
                        return false;
                    }
                    buf.str(line.substr(lastpos, pos));
                    int var_num;
                    buf >> var_num;
                    GVT_DEBUG(DBG_LOW,"    var_num: " << var_num);

                    // next three ints are the chunk dimensions
                    vector<int> dims;
                    GVT_DEBUG(DBG_LOW,"    dims: ");
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
                        GVT_DEBUG(DBG_LOW,d << " ");
                    }
                    this->sizes.push_back(dims);
                    

                    // next three ints are the chunk position offset
                    vector<int> offs;
                    GVT_DEBUG(DBG_LOW,"    offs: ");
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
                        if (d < this->min[i]) min[i] = d;
                        if ((d + dims[i]) > this->max[i]) max[i] = d + dims[i];
                        if ((d + dims[i]) > this->size[i]) size[i] = d + dims[i];
                        GVT_DEBUG(DBG_LOW, d << " ");
                    }
                    // special case for last int on line
                    {
                        buf.str(line.substr(pos + 1));
                        int d;
                        buf >> d;
                        offs.push_back(d);
                        if (d < this->min[2]) min[2] = d;
                        if ((d + dims[2]) > this->max[2]) max[2] = d + dims[2];
                        if ((d + dims[2]) > this->size[2]) size[2] = d + dims[2];
                        GVT_DEBUG(DBG_LOW, d << " ");
                    }

                    this->offsets.push_back(offs);

                    GVT::Data::box3D bb(
                            GVT::Math::Point4f((double)offs[0], (double)offs[1], (double)offs[2]),
                            GVT::Math::Point4f((double)offs[0] + dims[0], (double)offs[1] + dims[1], (float)offs[2] + dims[2])
                            );

                    GVT_DEBUG(DBG_SEVERE,"BBox " << bb);
                    GVT_DEBUG(DBG_SEVERE,"low " << offs[0]<< " " << offs[1] << " " << offs[2]);
                    
                    dom_bbox[domit++] = bb;

                    //DEBUG(cout << endl);
                }
            }

            return true;
        }

        template<>
        GVT::Domain::Domain*
        Dataset<GVT::Domain::VolumeDomain>::GetDomain(int id) {
            if (id < 0 || id >= files.size()) {
                GVT_DEBUG(DBG_LOW,"ERROR: invalid domain id '" << id << "' passed to GetDomain");
                return NULL;
            }

            // check domain cache first
            // XXX TODO remove GVT::Domain::VolumeDomain, make this a subclass
            map< int, GVT::Domain::VolumeDomain >::iterator it = dom_cache.find(id);
            if (it != dom_cache.end())
                return &(it->second);

            // if domain isn't in cache, create it
            vector<float> min, max;
            for (int i = 0; i < 3; ++i) {
                min.push_back((float) offsets[id][i]);
                max.push_back((float) (offsets[id][i] + sizes[id][i]));
            }
            dom_cache[id] = GVT::Domain::VolumeDomain(id, files[id], sizes[id], min, max); // XXX TODO fix when this made a subclass

            //cout << dom_cache[id] << endl;

            return &(dom_cache[id]);
        }

        ostream&
        operator<<(ostream& out, abstract_dataset const& ds) {
            //    out << "conf filename: " << ds.conf_filename << "\n";
            //    out << "    " << ds.files.size() << " files\n";
            //    out << "    " << ds.sizes.size() << " sizes\n";
            //    out << "    " << ds.offsets.size() << " offsets\n";
            //    out << "    min:  [ " << ds.min[0]  << ", " << ds.min[1]  << ", " << ds.min[2]  << " ]\n";
            //    out << "    max:  [ " << ds.max[0]  << ", " << ds.max[1]  << ", " << ds.max[2]  << " ]\n";
            //    out << "    size: [ " << ds.size[0] << ", " << ds.size[1] << ", " << ds.size[2] << " ]\n";    
            //    
            return out;
        }
    }
}