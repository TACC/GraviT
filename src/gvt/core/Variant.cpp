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

#include <gvt/core/Variant.h>

using namespace gvt::core;
using namespace gvt::core::math;

Variant::Variant() {}
Variant::Variant(int i) : coreData(i) {}
Variant::Variant(long l) : coreData(l) {}
Variant::Variant(float f) : coreData(f) {}
Variant::Variant(double d) : coreData(d) {}
Variant::Variant(bool b) : coreData(b) {}
Variant::Variant(unsigned long long ull) : coreData(ull) {}
Variant::Variant(String s) : coreData(s) {}
Variant::Variant(Uuid u) : coreData(u) {}
Variant::Variant(Vector3f v) : coreData(v) {}
Variant::Variant(Vector4f v) : coreData(v) {}
Variant::Variant(Point4f p) : coreData(p) {}

int Variant::toInteger() const { return boost::get<int>(coreData); }

long Variant::toLong() const { return boost::get<long>(coreData); }

float Variant::toFloat() const { return boost::get<float>(coreData); }

double Variant::toDouble() const { return boost::get<double>(coreData); }

bool Variant::toBoolean() const { return boost::get<bool>(coreData); }

unsigned long long Variant::toULongLong() const { return boost::get<unsigned long long>(coreData); }

String Variant::toString() const { return boost::get<String>(coreData); }

Uuid Variant::toUuid() const { return boost::get<Uuid>(coreData); }

Vector3f Variant::toVector3f() const { return boost::get<Vector3f>(coreData); }

Vector4f Variant::toVector4f() const { return boost::get<Vector4f>(coreData); }

Point4f Variant::toPoint4f() const { return boost::get<Point4f>(coreData); }

bool Variant::operator==(const Variant &v) const { return coreData == v.coreData; }

bool Variant::operator!=(const Variant &v) const { return !(coreData == v.coreData); }

namespace gvt {
namespace core {
std::ostream &operator<<(std::ostream &os, const Variant &v) { return os << v.coreData; }
}
}
