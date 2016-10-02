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
#ifndef GVT_CORE_DATABASE_NODE_H
#define GVT_CORE_DATABASE_NODE_H

#include "gvt/core/Types.h"
#include <gvt/core/Debug.h>
#include <gvt/render/data/primitives/BBox.h>

namespace gvt {
namespace core {
class DatabaseNode {

  Uuid p_uuid;
  String p_name;
  Uuid p_parent;
  Variant p_value;

public:
  DatabaseNode(String name, Variant value, Uuid uuid, Uuid parentUUID);

  Uuid UUID();
  String name();
  Uuid parentUUID();
  Variant value();

  void setUUID(Uuid uuid);
  void setName(String name);
  void setParentUUID(Uuid parentUUID);
  void setValue(Variant value);
  Vector<DatabaseNode *> getChildren();

  void propagateUpdate();
  explicit operator bool() const;



  static DatabaseNode *errNode;
};

class DBNodeH {
public:
  explicit DBNodeH(Uuid u = Uuid::null());
  Uuid UUID();
  String name();
  Uuid parentUUID();
  Variant value();

  void setUUID(Uuid uuid);
  void setName(String name);
  void setParentUUID(Uuid parentUUID);
  void setValue(Variant value);

  Vector<DBNodeH> getChildren();

  void propagateUpdate();
  DBNodeH deRef();
  void remove();

  DatabaseNode &getNode();

  DBNodeH operator[](const String &key);
  DBNodeH &operator+=(DBNodeH child);
  DBNodeH &operator=(Variant val);
  bool operator==(const Variant val);
  explicit operator bool() const;

  void connectValueChanged(const void *receiver, const char *method);
  void connectChildChanged(const void *receiver, const char *method);

private:
  Uuid _uuid;
};
}
}

#endif // GVT_CORE_DATABASE_NODE_H
