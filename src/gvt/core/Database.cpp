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
#include <iostream>
#include "gvt/core/Database.h"
#include "gvt/core/Debug.h"

using namespace gvt::core;

Database::Database() {}

Database::~Database() {
  for (Map<Uuid, DatabaseNode *>::iterator it = __nodes.begin(); it != __nodes.end(); ++it) {
    delete it->second;
  }
}

DatabaseNode *Database::getItem(Uuid uuid) { return __nodes[uuid]; }

void Database::setItem(DatabaseNode *node) {
  __nodes[node->UUID()] = node;
  addChild(node->parentUUID(), node);
}

void Database::setRoot(DatabaseNode *root) { __nodes[root->UUID()] = root; }

bool Database::hasNode(Uuid uuid) { return (__nodes.find(uuid) != __nodes.end()); }

bool Database::hasNode(DatabaseNode *node) { return (__nodes.find(node->UUID()) != __nodes.end()); }

ChildList &Database::getChildren(Uuid parent) { return __tree[parent]; }

void Database::addChild(Uuid parent, DatabaseNode *node) { __tree[parent].push_back(node); }

void Database::removeItem(Uuid uuid) {
  if (__nodes[uuid] != NULL) {
    DatabaseNode *cnode = getItem(uuid);

    GVT_DEBUG(DBG_LOW, "Removing item:");
    GVT_DEBUG_CODE(DBG_LOW, print(uuid, 0, std::cerr));

    ChildList *children = &__tree[uuid];
    ChildList::iterator it;

    GVT_DEBUG(DBG_LOW, "removing children:");
    GVT_DEBUG_CODE(DBG_LOW,
                   for (it = children->begin(); it != children->end(); it++) print((*it)->UUID(), 0, std::cerr););
    GVT_DEBUG(DBG_LOW, "");

    int numChildren = children->size();
    for (int i = 0; i < numChildren; i++) {
      removeItem((children->at(0))->UUID());
    }
    children = &__tree[cnode->parentUUID()];
    GVT_DEBUG_CODE(DBG_LOW, for (it = children->begin(); it != children->end(); it++) std::cerr
                                << "tree item: " << (*it)->UUID().toString() << std::endl;);
    for (it = children->begin(); it != children->end(); ++it) {
      if ((*it)->UUID() == uuid) break;
    }
    Uuid puid = cnode->parentUUID();
    GVT_DEBUG(DBG_LOW, "found tree item to remove from parent: " << (*it)->name() << " " << (*it)->UUID().toString());
    if (it != children->end()) children->erase(it);
    __nodes.erase(uuid);
    delete cnode;
  } else {
    GVT_DEBUG(DBG_MODERATE, "ERROR: Could not find item to remove from database : " << uuid.toString());
  }
}

DatabaseNode *Database::getChildByName(Uuid parent, String name) {
  ChildList children = __tree[parent];
  for (ChildList::iterator it = children.begin(); it != children.end(); ++it) {
    if ((*it)->name() == name) return (*it);
  }
  return NULL;
}

void Database::print(const Uuid &parent, const int depth, std::ostream &os) {

  DatabaseNode *pnode = this->getItem(parent);
  if (!pnode) {
    GVT_DEBUG(DBG_MODERATE, "Database::print - node not found: " << parent.toString());
    return;
  }
  std::string offset = "";
  for (int i = 0; i < depth; i++) offset += "-";
  os << offset << pnode->UUID().toString() << " : " << pnode->name() << " : " << pnode->value() << std::endl;
  offset += "-";
  ChildList children = __tree[parent];
  for (ChildList::iterator it = children.begin(); it != children.end(); ++it) {
    DatabaseNode *node = (*it);
    os << offset << node->UUID().toString() << " : " << node->name() << " : " << node->value() << std::endl;
  }
}

void Database::printTree(const Uuid &parent, const int depth, std::ostream &os) {
  DatabaseNode *pnode = this->getItem(parent);
  if (!pnode) {
    GVT_DEBUG(DBG_ALWAYS, "Database::printTree - node not found: " << parent.toString());
    return;
  }
  std::string offset = "";
  for (int i = 0; i < depth; i++) offset += "-";
  offset += "|";
  os << offset << pnode->UUID().toString() << " : " << pnode->name() << " : " << pnode->value() << std::endl;
  ChildList children = __tree[parent];
  for (ChildList::iterator it = children.begin(); it != children.end(); ++it) {
    DatabaseNode *node = (*it);
    printTree(node->UUID(), depth + 1, os);
  }
}

Variant Database::getValue(Uuid id) {
  Variant val;
  DatabaseNode *node = getItem(id);
  if (node) val = node->value();
  return val;
}

void Database::setValue(Uuid id, Variant val) {
  DatabaseNode *node = getItem(id);
  if (node) node->setValue(val);
}

Variant Database::getChildValue(Uuid parent, String child) {
  Variant val;
  DatabaseNode *node = getChildByName(parent, child);
  if (node) val = node->value();
  return val;
}

void Database::setChildValue(Uuid parent, String child, Variant value) {
  DatabaseNode *node = getChildByName(parent, child);
  if (node) node->setValue(value);
}
