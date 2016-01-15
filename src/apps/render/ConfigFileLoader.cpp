/* =======================================================================================
   This file is released as part of GraviT - scalable, platform independent ray tracing
   tacc.github.io/GraviT

   Copyright 2013-2015 Texas Advanced Computing Center, The University of Texas at Austin
   All rights reserved.

   Licensed under the BSD 3-Clause License, (the "License"); you may not use this file
   except in compliance with the License.
   A copy of the License is included with this software in the file LICENSE.
   If your copy does not contain the License, you may obtain a copy of the License at:

       http://opensource.org/licenses/BSD-3-Clause

   Unless required by applicable law or agreed to in writing, software distributed under
   the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
   KIND, either express or implied.
   See the License for the specific language governing permissions and limitations under
   limitations under the License.

   GraviT is funded in part by the US National Science Foundation under awards ACI-1339863,
   ACI-1339881 and ACI-1339840
   ======================================================================================= */
/*
 * File:   ConfigFileLoader.cpp
 * Author: jbarbosa
 *
 * Created on January 21, 2015, 12:15 PM
 */

#include "ConfigFileLoader.h"

#include <gvt/render/data/domain/reader/ObjReader.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <boost/regex.h>
#include <boost/regex.hpp>
#include <map>

using namespace gvtapps::render;

namespace gvtapps {
namespace render {
/// split a string into a set of substrings based on the supplied delimiter.
/**
  \param s the string to split
  \param delim a char delimiter
  \param elems a string vector on which the substrings are placed
*/
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
}
} // namespace render} namespace gvtapps}

/// constructor
/**
    \param filename configuration file
*/
ConfigFileLoader::ConfigFileLoader(const std::string filename) {

  GVT_ASSERT(filename.size() > 0, "Error filename not specified");
  GVT_DEBUG(DBG_ALWAYS, "Loading file : " << filename);
  std::fstream file;
  file.open(filename.c_str());
  GVT_ASSERT(file.good(), "Error loading config file " << filename);

  while (file.good()) {
    std::string line;
    std::getline(file, line);
    if (line.find("#") == 0) continue;

    std::vector<std::string> elems;
    split(line, ' ', elems);

    if (elems.empty()) continue;

    if (elems[0] == "F") {
      scene.camera.setFilmSize(std::atoi(elems[1].c_str()), std::atoi(elems[2].c_str()));
    } else if (elems[0] == "C") {
      gvt::core::math::Vector4f pos, look, up, focus;
      float fov = std::atof(elems[10].c_str()) * M_PI / 180.0;
      pos[0] = std::atof(elems[1].c_str());
      pos[1] = std::atof(elems[2].c_str());
      pos[2] = std::atof(elems[3].c_str());
      pos[3] = 1.f;
      look[0] = std::atof(elems[4].c_str());
      look[1] = std::atof(elems[5].c_str());
      look[2] = std::atof(elems[6].c_str());
      look[3] = 0.f;
      up[0] = std::atof(elems[7].c_str());
      up[1] = std::atof(elems[8].c_str());
      up[2] = std::atof(elems[9].c_str());
      up[3] = 0.f;

      scene.camera.setLook(pos, look, up);
      scene.camera.setFOV(fov);
      scene.camera.setAspectRatio((float)scene.camera.filmsize[0] / (float)scene.camera.filmsize[1]);

    } else if (elems[0] == "G") {
      GVT_DEBUG(DBG_ALWAYS, "Geometry file" << elems[1]);

      gvt::render::data::domain::GeometryDomain *domain = NULL;

      if (elems[1].find(".obj") < elems[1].size()) {
        GVT_DEBUG(DBG_ALWAYS, "Found obj file : " << elems[1].find(".obj"));

        gvt::render::data::primitives::Mesh *mesh;

        std::map<std::string, gvt::render::data::primitives::Mesh *>::iterator meshIt = scene.objMeshes.find(elems[1]);

        if (meshIt != scene.objMeshes.end()) {
          mesh = meshIt->second;
        } else {
          gvt::render::data::domain::reader::ObjReader objReader(elems[1]);
          mesh = objReader.getMesh();
          scene.objMeshes[elems[1]] = mesh;
        }

        scene.domainSet.push_back(domain = new gvt::render::data::domain::GeometryDomain(mesh));

        gvt::core::math::Vector4f t;
        t[0] = std::atof(elems[2].c_str());
        t[1] = std::atof(elems[3].c_str());
        t[2] = std::atof(elems[4].c_str());

        GVT_DEBUG(DBG_ALWAYS, "Translate vector : \n" << t);

        if (t.length() > 0.0) domain->translate(t);

        t[0] = std::atof(elems[5].c_str());
        t[1] = std::atof(elems[6].c_str());
        t[2] = std::atof(elems[7].c_str());

        if (t.length() > 0.0) domain->rotate(t);

        t[0] = std::atof(elems[8].c_str());
        t[1] = std::atof(elems[9].c_str());
        t[2] = std::atof(elems[10].c_str());

        GVT_DEBUG(DBG_ALWAYS, "Scale vector : \n" << t);

        if (t.length() > 0.0) domain->scale(t);

        GVT_DEBUG(DBG_ALWAYS, "Aff. m : \n" << domain->m);
        GVT_DEBUG(DBG_ALWAYS, "Aff. minv : \n" << domain->minv);
        GVT_DEBUG(DBG_ALWAYS, "BB : \n" << domain->getWorldBoundingBox());
        GVT_DEBUG(DBG_ALWAYS, "BB : \n" << domain->boundingBox);
      }
      if (elems[1].find(".ply") < elems[1].size()) {
        GVT_DEBUG(DBG_ALWAYS, "Found ply file : " << elems[1].find(".ply"));
      }
    } else if (elems[0] == "LP") {
      gvt::core::math::Vector4f pos, color;
      pos[0] = std::atof(elems[1].c_str());
      pos[1] = std::atof(elems[2].c_str());
      pos[2] = std::atof(elems[3].c_str());
      pos[3] = 1.f;
      color[0] = std::atof(elems[4].c_str());
      color[1] = std::atof(elems[5].c_str());
      color[2] = std::atof(elems[6].c_str());
      color[3] = 1.f;
      scene.lightSet.push_back(new gvt::render::data::scene::PointLight(pos, color));
    } else if (elems[0] == "LA") {
      gvt::core::math::Vector4f pos[2], color;
      pos[0][0] = std::atof(elems[1].c_str());
      pos[0][1] = std::atof(elems[2].c_str());
      pos[0][2] = std::atof(elems[3].c_str());
      pos[0][3] = 1.f;
      pos[1][0] = std::atof(elems[4].c_str());
      pos[1][1] = std::atof(elems[5].c_str());
      pos[1][2] = std::atof(elems[6].c_str());
      pos[1][3] = 1.f;
      color[0] = std::atof(elems[7].c_str());
      color[1] = std::atof(elems[8].c_str());
      color[2] = std::atof(elems[9].c_str());
      color[3] = 1.f;
      GVT_DEBUG(DBG_ALWAYS, "Light area not implemented");
    } else if (elems[0] == "RT") {
      GVT_DEBUG(DBG_ALWAYS, "option RT not supported");
    } else if (elems[0] == "ST") {
      GVT_DEBUG(DBG_ALWAYS, "Light area not implemented");
    } else if (elems[0] == "DT") {
      if (elems[1] == "OPTIX") domain_type = 1;
      if (elems[1] == "EMBREE") domain_type = 2;
    } else if (elems[0] == "ST") {
      if (elems[1] == "DOMAIN") scheduler_type = 1;
      if (elems[1] == "HYBRID") scheduler_type = 2;
    } else if (elems[0] == "AT") {
      if (elems[1] == "BVH") accel_type = BVH;
    } else {
      GVT_DEBUG(DBG_LOW, "Invalid option");
    }
  }
}

ConfigFileLoader::ConfigFileLoader(const ConfigFileLoader &orig) {}

ConfigFileLoader::~ConfigFileLoader() {}
