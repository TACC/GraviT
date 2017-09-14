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
#ifndef GVT_CORE_UUID_H
#define GVT_CORE_UUID_H


#include <ostream>
#include <atomic>
#include <gvt/core/cntx/identifier.h>

namespace gvt {
namespace core {
/// unique identifier used to tag nodes in the context database
/**
* \sa CoreContext, Database, DatabaseNode
*/



class Uuid : public cntx::details::identifier {
public:
  Uuid() : identifier(all_ranks, gen()) {}

  void nullify() { this->id = invalid_id; }

  bool isNull() const { return this->id == invalid_id; }

  std::string toString() const { return std::to_string(getid()); }

  bool operator==(const Uuid &u) const { return getid() == u.getid(); }
  bool operator!=(const Uuid &u) const { return getid() != u.getid(); }

  bool operator>(const Uuid &u) const { return getid() > u.getid(); }

  bool operator<(const Uuid &u) const { return getid() < u.getid(); }

  friend std::ostream &operator<<(std::ostream &, const Uuid &);
  static Uuid null();

protected:
//  boost::uuids::uuid uuid;

private:
  static std::atomic<unsigned long> idinc;

  static unsigned long gen() {
    return (unsigned long) ( Uuid::idinc++);
  }

//  static boost::uuids::random_generator gen;
};
}
}
#endif // GVT_CORE_UUID_H
