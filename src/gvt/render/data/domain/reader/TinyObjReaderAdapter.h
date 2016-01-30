/* =======================================================================================
This file is released as part of GraviT - scalable, platform independent ray
tracing
tacc.github.io/GraviT

Copyright 2013-2015 Texas Advanced Computing Center, The University of Texas
at Austin
All rights reserved.

Licensed under the BSD 3-Clause License, (the "License"); you may not use
this file
except in compliance with the License.
A copy of the License is included with this software in the file LICENSE.
If your copy does not contain the License, you may obtain a copy of the
License at:

   http://opensource.org/licenses/BSD-3-Clause

Unless required by applicable law or agreed to in writing, software
distributed under
the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY
KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under
limitations under the License.

GraviT is funded in part by the US National Science Foundation under awards
ACI-1339863,
ACI-1339881 and ACI-1339840
=======================================================================================
*/

/*
* File:   TinyObjReaderAdapter.h
* Author: vchen
*
* Created on April 22, 2014, 10:24 AM
*/

#ifndef GVT_RENDER_DATA_DOMAIN_TINY_OBJ_READER_ADAPTER_H
#define GVT_RENDER_DATA_DOMAIN_TINY_OBJ_READER_ADAPTER_H

#include <gvt/core/Math.h>
#include <gvt/render/data/Primitives.h>

namespace gvt {
namespace render {
namespace data {
namespace domain {
namespace reader {

class TinyObjReaderAdapter {
public:
  
  TinyObjReaderAdapter(const std::string filename, const std::string materialsBasePath);
  virtual ~TinyObjReaderAdapter();

  int getMeshCount() {return meshCount;}

  gvt::render::data::primitives::Mesh **getMesh() { return objMeshes; }
  gvt::render::data::primitives::Box3D**getBox3D() { return meshbboxes; }

protected:
  int meshCount = 0;
  gvt::render::data::primitives::Mesh **objMeshes;
  gvt::render::data::primitives::Box3D **meshbboxes;
};

}//reader
}//domain
}//data
}//render
}//gvt

#endif /* GVT_RENDER_DATA_DOMAIN_TINY_OBJ_READER_ADAPTER_H */
