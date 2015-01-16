//
//  OptixDataset.C
//
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include <Backend/Optix/Domain/OptixDomain.h>
#include <Backend/Optix/Dataset/OptixDataset.h>

namespace GVT {

namespace Dataset {
    
bool OptixDataset::init() {
  float min[3];
  float max[3];
  int size[3];
  vector<vector<int> > sizes;
  vector<vector<int> > offsets;
  std::cout << "Opened file" << std::endl;
  GVT_ASSERT(conf_filename.size() > 0,
             "configuration file for dataset not set : " << conf_filename);

  fstream conf;
  conf.open(conf_filename.c_str(), fstream::in);
  GVT_ASSERT(conf.is_open(), "ERROR: Could not open file '" << conf_filename
                                                            << "'");
  ;
  // pull absolute path from conf_filename to prepend to chunk filenames
  size_t ptr = conf_filename.find_last_of("/\\");
  string conf_dir = conf_filename.substr(0, ptr);

  GVT_DEBUG(DBG_LOW, "opened file '" << conf_filename << "' with dir '"
                                     << conf_dir << "'");

  // read chunk filenames and layout data
  while (conf.good()) {
    string line;
    getline(conf, line);
    if (line.size() > 0 && line.substr(0, 1) != "#") {
      GVT_DEBUG(DBG_LOW, "got line: '" << line << "'");
      line += " ";
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
        GVT_ASSERT(pos != string::npos, "ERROR: didn't find translate "
                                            << i << " in chunk line: '" << line
                                            << "'");
        buf.str(line.substr(lastpos, pos));
        buf >> t[i];
      }

      GVT::Math::Vector4f r(0.f, 0.f, 0.f, 0.f);
      for (int i = 0; i < 4; ++i) {
        lastpos = pos + 1;
        pos = line.find_first_of(" ", lastpos);
        GVT_ASSERT(pos != string::npos, "ERROR: didn't find rotate "
                                            << i << " in chunk line: '" << line
                                            << "' (if i = 3, probably needs a "
                                               "extra space at the end of the "
                                               "line)");
        buf.str(line.substr(lastpos, pos));
        buf >> r[i];
      }

      GVT::Math::AffineTransformMatrix<float> m =
          GVT::Math::AffineTransformMatrix<float>::createTranslation(t[0], t[1],
                                                                     t[2]);

      GVT::Domain::OptixDomain* dom = new GVT::Domain::OptixDomain(file, m);
      GVT_DEBUG(DBG_LOW,
                "Domain bounding box : " << dom->getWorldBoundingBox());
      this->addDomain(dom);
    } else {
      GVT_DEBUG(DBG_LOW, "Ignored line : '" << line << "'");
    }
  }
  return true;
}

}  // namespace Domain

}  // namespace GVT
