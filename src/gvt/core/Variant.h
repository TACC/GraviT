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
#ifndef GVT_CORE_VARIANT_H
#define GVT_CORE_VARIANT_H

#include <gvt/core/Math.h>
#include <gvt/core/String.h>
#include <gvt/core/Uuid.h>

#include <boost/variant.hpp>

#include <ostream>

namespace gvt {
namespace core {
/// mutable container class for datatypes used in the context database
/**
* \sa CoreContext, Database, DatabaseNode
*/
class Variant {
public:
  Variant();
  Variant(int);
  Variant(long);
  Variant(float);
  Variant(double);
  Variant(bool);
  Variant(unsigned long long);
  Variant(String);
  Variant(Uuid);
  Variant(gvt::core::math::Vector3f);
  Variant(gvt::core::math::Vector4f);
  Variant(gvt::core::math::Point4f);

  int toInteger() const;
  long toLong() const;
  float toFloat() const;
  double toDouble() const;
  bool toBoolean() const;
  String toString() const;
  Uuid toUuid() const;
  unsigned long long toULongLong() const;
  gvt::core::math::Vector3f toVector3f() const;
  gvt::core::math::Vector4f toVector4f() const;
  gvt::core::math::Point4f toPoint4f() const;

  bool operator==(const Variant &) const;
  bool operator!=(const Variant &) const;

  friend std::ostream &operator<<(std::ostream &, const Variant &);

protected:
  boost::variant<int, long, float, double, bool, unsigned long long, String, Uuid, gvt::core::math::Vector3f,
                 gvt::core::math::Vector4f, gvt::core::math::Point4f> coreData;
};
}
}

#endif // GVT_CORE_VARIANT_H
