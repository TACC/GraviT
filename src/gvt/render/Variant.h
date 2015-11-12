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

#ifndef GVT_RENDER_VARIANT_H
#define GVT_RENDER_VARIANT_H

#include <gvt/core/Variant.h>
#include <gvt/render/data/Primitives.h>

#include <ostream>

namespace gvt {
   namespace render {
      /// mutable container class for rendering-specific datatypes used in the context database
      /**
      * \sa CoreContext, Database, DatabaseNode
      */
      class Variant : public gvt::core::Variant
      {
      public:
         Variant();
         Variant(int);
         Variant(long);
         Variant(float);
         Variant(double);
         Variant(bool);
         Variant(gvt::core::String);
         Variant(gvt::core::Uuid);
         Variant(gvt::core::math::Vector3f);
         Variant(gvt::core::math::Vector4f);
         Variant(gvt::core::math::Point4f);
         Variant(gvt::core::math::AffineTransformMatrix<float>*);
         Variant(gvt::core::math::Matrix3f*);
         Variant(gvt::render::data::primitives::Mesh*);
         Variant(gvt::render::data::primitives::Box3D*);

         gvt::render::data::primitives::Mesh* toMesh() const;
         gvt::render::data::primitives::Box3D* toBox3D() const;

         bool operator==(const Variant&) const;
         bool operator!=(const Variant&) const;

         friend std::ostream& operator<<(std::ostream&, const Variant&);

      protected:
         bool hasRenderData;
         boost::variant<gvt::render::data::primitives::Mesh*,
                        gvt::render::data::primitives::Box3D*> renderData;
      };
   }
}

#endif // GVT_RENDER_VARIANT_H