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

#include <gvt/render/data/reader/ObjReader.h>

#include <boost/regex.h>
#include <boost/regex.hpp>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <gvt/render/RenderContext.h>
#include <gvt/core/Math.h>


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


  gvt::render::RenderContext *cntxt = gvt::render::RenderContext::instance();
  /*
   * Assumes that required base nodes are created
   */
  gvt::core::DBNodeH root = cntxt->getRootNode();

  std::map<std::string, gvt::render::data::primitives::Mesh *> meshes;
  int domainCount=0;

  while (file.good()) {
    std::string line;
    std::getline(file, line);
    if (line.find("#") == 0) continue;

    std::vector<std::string> elems;
    split(line, ' ', elems);

    if (elems.empty()) continue;

    if (elems[0] == "F") {

    gvt::core::DBNodeH filmNode = root["Film"];
    filmNode["width"] = std::atoi(elems[1].c_str());
    filmNode["height"] =  std::atoi(elems[2].c_str());

    } else if (elems[0] == "C") {
      glm::vec3 pos, look, up, focus;
      float fov = std::atof(elems[10].c_str()) * M_PI / 180.0;
      pos[0] = std::atof(elems[1].c_str());
      pos[1] = std::atof(elems[2].c_str());
      pos[2] = std::atof(elems[3].c_str());
      look[0] = std::atof(elems[4].c_str());
      look[1] = std::atof(elems[5].c_str());
      look[2] = std::atof(elems[6].c_str());
      up[0] = std::atof(elems[7].c_str());
      up[1] = std::atof(elems[8].c_str());
      up[2] = std::atof(elems[9].c_str());

      // camera
      gvt::core::DBNodeH _camNode = root["Camera"];
      _camNode["eyePoint"] = pos;
      _camNode["focus"] = look;
      _camNode["upVector"] = up;
      _camNode["fov"] = (float)(45.0 * M_PI / 180.0); // TODO

    } else if (elems[0] == "G") {
      GVT_DEBUG(DBG_ALWAYS, "Geometry file" << elems[1]);

     // gvt::render::data::domain::GeometryDomain *domain = NULL;
      gvt::render::data::primitives::Mesh *mesh;

      glm::mat4* m = new glm::mat4(1.f);
      glm::mat4* minv= new glm::mat4(1.f);
      glm::mat3* normi = new glm::mat3(1.f);

      if (elems[1].find(".obj") < elems[1].size()) {
        GVT_DEBUG(DBG_ALWAYS, "Found obj file : " << elems[1].find(".obj"));


      std::map<std::string, gvt::render::data::primitives::Mesh *>::iterator meshIt = meshes.find(elems[1]);

      if (meshIt != meshes.end()) {
        mesh = meshIt->second;
      } else {
        gvt::render::data::domain::reader::ObjReader objReader(elems[1]);
        mesh = objReader.getMesh();
        meshes[elems[1]] = mesh;
      }

      mesh->generateNormals();

     // domain = new gvt::render::data::domain::GeometryDomain(mesh);

        glm::vec3 t;
        t[0] = std::atof(elems[2].c_str());
        t[1] = std::atof(elems[3].c_str());
        t[2] = std::atof(elems[4].c_str());

        GVT_DEBUG(DBG_ALWAYS, "Translate vector : \n" << t);

        if (glm::length(t) > 0.0) {
           *m = glm::translate(*m, t);
           *minv = glm::inverse(*m);
           *normi =  glm::transpose(glm::inverse(glm::mat3(*m)));
        }

        t[0] = std::atof(elems[5].c_str());
        t[1] = std::atof(elems[6].c_str());
        t[2] = std::atof(elems[7].c_str());

        if (glm::length(t) > 0.0) {

        	 glm::mat4 mAA = glm::rotate(*m, t[0], glm::vec3(1, 0, 0)) *
        	                  glm::rotate(*m, t[1], glm::vec3(0, 1, 0)) *
        	                  glm::rotate(*m, t[2], glm::vec3(0, 0, 1));

             *minv = glm::inverse(*m);
             *normi =  glm::transpose(glm::inverse(glm::mat3(*m)));
        }

        t[0] = std::atof(elems[8].c_str());
        t[1] = std::atof(elems[9].c_str());
        t[2] = std::atof(elems[10].c_str());

        GVT_DEBUG(DBG_ALWAYS, "Scale vector : \n" << t);

        if (glm::length(t) > 0.0) {
                   *m = glm::scale(*m, t);
                   *minv = glm::inverse(*m);
                   *normi =  glm::transpose(glm::inverse(glm::mat3(*m)));
                }
      }
      if (elems[1].find(".ply") < elems[1].size()) {
        GVT_DEBUG(DBG_ALWAYS, "Found ply file : " << elems[1].find(".ply"));
      }

		gvt::core::DBNodeH dataNodes = root["Data"];
		gvt::core::DBNodeH instNodes = root["Instances"];

		//gvt::render::data::primitives::Mesh *mesh = domain->getMesh();

		gvt::core::DBNodeH meshNode = cntxt->createNodeFromType("Mesh",
				filename.c_str(), dataNodes.UUID());

		meshNode["file"] = filename;
		gvt::render::data::primitives::Box3D *bbox = mesh->getBoundingBox();
		meshNode["bbox"] = (unsigned long long) bbox;
		meshNode["ptr"] = (unsigned long long) mesh;

		// add instance
		gvt::core::DBNodeH instnode = cntxt->createNodeFromType("Instance",
				"inst", instNodes.UUID());
		gvt::render::data::primitives::Box3D *mbox =
				(gvt::render::data::primitives::Box3D *) meshNode["bbox"].value().toULongLong();
		instnode["id"] = domainCount++;
		instnode["meshRef"] = meshNode.UUID();

		instnode["mat"] = (unsigned long long) m;
		instnode["matInv"] = (unsigned long long) minv;
		instnode["normi"] = (unsigned long long) normi;

		auto il = glm::vec3((*m) * glm::vec4(mbox->bounds_min, 1.f));
		auto ih = glm::vec3((*m) * glm::vec4(mbox->bounds_max, 1.f));
		gvt::render::data::primitives::Box3D *ibox =
				new gvt::render::data::primitives::Box3D(il, ih);
		instnode["bbox"] = (unsigned long long) ibox;
		instnode["centroid"] = ibox->centroid();



    } else if (elems[0] == "LP") {
      glm::vec3 pos, color;
      pos[0] = std::atof(elems[1].c_str());
      pos[1] = std::atof(elems[2].c_str());
      pos[2] = std::atof(elems[3].c_str());
      color[0] = std::atof(elems[4].c_str());
      color[1] = std::atof(elems[5].c_str());
      color[2] = std::atof(elems[6].c_str());


      gvt::core::DBNodeH lightNode = cntxt->createNodeFromType(
    		  "PointLight", "PointLight", root["Lights"].UUID());

        gvt::render::data::scene::PointLight *lp =
        		new gvt::render::data::scene::PointLight(pos, color);
        lightNode["position"] = glm::vec3(lp->position);
        lightNode["color"] = glm::vec3(lp->color);

    } else if (elems[0] == "LA") {
      glm::vec3 pos, normal, color;
      float width,height;
      pos[0] = std::atof(elems[1].c_str());
      pos[1] = std::atof(elems[2].c_str());
      pos[2] = std::atof(elems[3].c_str());
      normal[0] = std::atof(elems[4].c_str());
      normal[1] = std::atof(elems[5].c_str());
      normal[2] = std::atof(elems[6].c_str());
      width = std::atof(elems[7].c_str());
      height = std::atof(elems[8].c_str());
      color[0] = std::atof(elems[9].c_str());
      color[1] = std::atof(elems[10].c_str());
      color[2] = std::atof(elems[11].c_str());

      gvt::core::DBNodeH lightNode = cntxt->createNodeFromType(
                  "AreaLight", "AreaLight", root["Lights"].UUID());


      lightNode["position"] = pos;
      lightNode["normal"] = normal;
      lightNode["width"] = width;
      lightNode["height"] = height;
      lightNode["color"] = color;


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
