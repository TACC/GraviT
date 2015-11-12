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


#include <gvt/render/Variant.h>

using namespace gvt::core;
using namespace gvt::core::math;
using namespace gvt::render;
using namespace gvt::render::data::primitives;


gvt::render::Variant::Variant(): gvt::core::Variant() {}
gvt::render::Variant::Variant(int i): gvt::core::Variant(i), hasRenderData(false) {}
gvt::render::Variant::Variant(long l): gvt::core::Variant(l), hasRenderData(false) {}
gvt::render::Variant::Variant(float f): gvt::core::Variant(f), hasRenderData(false) {}
gvt::render::Variant::Variant(double d): gvt::core::Variant(d), hasRenderData(false) {}
gvt::render::Variant::Variant(bool b): gvt::core::Variant(b), hasRenderData(false) {}
gvt::render::Variant::Variant(String s): gvt::core::Variant(s), hasRenderData(false) {}
gvt::render::Variant::Variant(Uuid u): gvt::core::Variant(u), hasRenderData(false) {}
gvt::render::Variant::Variant(Vector3f v): gvt::core::Variant(v), hasRenderData(false) {}
gvt::render::Variant::Variant(Vector4f v): gvt::core::Variant(v), hasRenderData(false) {}
gvt::render::Variant::Variant(Point4f p): gvt::core::Variant(p), hasRenderData(false) {}
gvt::render::Variant::Variant(AffineTransformMatrix<float>* atm): gvt::core::Variant(atm), hasRenderData(false) {}
gvt::render::Variant::Variant(Matrix3f* m): gvt::core::Variant(m), hasRenderData(false) {}
gvt::render::Variant::Variant(Mesh* m): renderData(m), hasRenderData(true) {}
gvt::render::Variant::Variant(Box3D* b): renderData(b), hasRenderData(true) {}


Mesh* gvt::render::Variant::toMesh() const
{ 
   return boost::get<Mesh*>(renderData); 
}

Box3D* gvt::render::Variant::toBox3D() const
{
   return boost::get<Box3D*>(renderData);
}


bool gvt::render::Variant::operator==(const gvt::render::Variant& v) const
{
   return hasRenderData ? (renderData == v.renderData) : (coreData == v.coreData);
}

bool gvt::render::Variant::operator!=(const gvt::render::Variant& v) const 
{
   return hasRenderData ? (renderData != v.renderData) : (coreData != v.coreData);
}

namespace gvt {
   namespace render {
      std::ostream& operator<<(std::ostream& os, const Variant& v)
      {
         if (v.hasRenderData) os << v.renderData;
         else os << v.coreData;
         return os;
      }
   }
}