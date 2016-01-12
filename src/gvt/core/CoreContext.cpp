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
#include "gvt/core/CoreContext.h"
#include "gvt/core/Debug.h"

using namespace gvt::core;

CoreContext *CoreContext::__singleton = nullptr;

CoreContext::CoreContext() {
  __database = new Database();
  DatabaseNode *root = new DatabaseNode(String("GraviT"), String("GVT ROOT"), Uuid(), Uuid::null());
  __database->setRoot(root);
  __rootNode = DBNodeH(root->UUID());
}

CoreContext::~CoreContext() { delete __database; }

CoreContext *CoreContext::instance() {
  if (__singleton == nullptr) {
    __singleton = new CoreContext();
  }
  return static_cast<CoreContext *>(__singleton);
}

DBNodeH CoreContext::getNode(Uuid node) {
  DatabaseNode *n = __database->getItem(node);
  if (n)
    return DBNodeH(n->UUID());
  else
    return DBNodeH();
}

DBNodeH CoreContext::createNode(String name, Variant val, Uuid parent) {
  DatabaseNode *np = new DatabaseNode(name, val, Uuid(), parent);
  __database->setItem(np);
  GVT_DEBUG(DBG_LOW, "createNode: " << name << " " << np->UUID().toString());
  return DBNodeH(np->UUID());
}

DBNodeH CoreContext::createNodeFromType(String type, Uuid parent) { return createNodeFromType(type, type, parent); }

DBNodeH CoreContext::createNodeFromType(String type) { return createNodeFromType(type, type); }

DBNodeH CoreContext::createNodeFromType(String type, String name, Uuid parent) {
  DBNodeH n = createNode(type, name, parent);

  // TODO - make these for GraviT

  return n;
}
