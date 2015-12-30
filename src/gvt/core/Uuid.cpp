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

#include <gvt/core/Uuid.h>
#ifdef __USE_TAU
#include <TAU.h>
#endif

using namespace gvt::core;

  boost::uuids::random_generator Uuid::gen ;


Uuid Uuid::null()
{
   Uuid u;
   u.nullify();
   return u;
}

/*Uuid::Uuid()
{
#ifdef __USE_TAU
	TAU_START("UUid.cpp::boost::uuids::uuid(boost::uuids::random_generator()())");
#endif
	uuid = boost::uuids::uuid(boost::uuids::random_generator()());
#ifdef __USE_TAU
	TAU_START("UUid.cpp::boost::uuids::uuid(boost::uuids::random_generator()())");
#endif
}*/

/*Uuid::~Uuid()
{

}*/

//void Uuid::nullify()
//{
//	uuid = boost::uuids::nil_uuid();
//}
//
//bool Uuid::isNull() const
//{
//	return uuid == boost::uuids::nil_uuid();
//}
//
//String Uuid::toString() const
//{
//	return boost::uuids::to_string(uuid);
//}

//bool Uuid::operator==(const Uuid& u) const
//{
//	return uuid == u.uuid;
//}

//bool Uuid::operator!=(const Uuid& u) const
//{
//	return uuid != u.uuid;
//}

//bool Uuid::operator>(const Uuid& u) const
//{
//	return uuid > u.uuid;
//}

//bool Uuid::operator<(const Uuid& u) const
//{
//	return uuid < u.uuid;
//}

namespace gvt {
	namespace core {
      std::ostream& operator<<(std::ostream& os, const Uuid& u)
      {
         return os << u.uuid;
      }

	}
}
