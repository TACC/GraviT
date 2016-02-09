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
 * File:   convert.h
 * Author: jbarbosa
 *
 * Created on March 10, 2014, 12:43 PM
 */

#ifndef GVT_CORE_DATA_TRANSFORM_H
#define GVT_CORE_DATA_TRANSFORM_H

#include <gvt/core/Debug.h>

/// template for data transforms from source to destination format
/**
    template for transforming data between formats, typically used to transform work elements
    (e.g. actors, data) between GraviT internal representation and the format required for a
    particular work engine (e.g. Manta, Embree, Optix)

   explicit function template specifications must be defined in the same namespace as the primary
   template rather than put all transforms into gvt::core::data, making primary template a macro
   that can be included at the top of each specialization inside its resepctive namespace
*/

#define GVT_TRANSFORM_TEMPLATE                                                                                         \
  template <typename SRC_FORMAT, typename DST_FORMAT> struct transform_impl {                                          \
    static inline DST_FORMAT transform(const SRC_FORMAT &src) {                                                        \
      GVT_WARNING(DBG_ALWAYS, "CONVERSION NOT IMPLEMENTED : TRYING DYNAMIC CAST");                                     \
      return dynamic_cast<DST_FORMAT>(src);                                                                            \
    }                                                                                                                  \
  };                                                                                                                   \
                                                                                                                       \
  template <typename SRC_FORMAT, typename DST_FORMAT> inline DST_FORMAT transform(const SRC_FORMAT &param) {           \
    return transform_impl<SRC_FORMAT, DST_FORMAT>::transform(param);                                                   \
  }

#endif /* GVT_CORE_DATA_TRANSFORM_H */
