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
#include "gvt/core/Database.h"
#include "gvt/core/Debug.h"
#include <iostream>

using namespace gvt::core;

Database::Database() {}

Database::~Database() {
  for (Map<Uuid, DatabaseNode *>::iterator it = __nodes.begin(); it != __nodes.end(); ++it) {
    delete it->second;
  }
}

DatabaseNode *Database::getItem(Uuid uuid) { return __nodes[uuid]; }

DatabaseNode *Database::getParentNode(String parentName) { return getItem(__parents[parentName]); }

void Database::setItem(DatabaseNode *node) {
  __nodes[node->UUID()] = node;
  addChild(node->parentUUID(), node);
}

void Database::setRoot(DatabaseNode *root) { __nodes[root->UUID()] = root; }

bool Database::hasNode(Uuid uuid) { return (__nodes.find(uuid) != __nodes.end()); }

bool Database::hasNode(DatabaseNode *node) { return (__nodes.find(node->UUID()) != __nodes.end()); }

bool Database::hasParentNode(String parentName) { return (__parents.find(parentName) != __parents.end()); }

ChildList &Database::getChildren(Uuid parent) { return __tree[parent]; }

//add children and keep track of non-leafs nodes by name
void Database::addChild(Uuid parent, DatabaseNode *node) {
	if (parent == Uuid::null())
		return;

	DatabaseNode *parentNode = getItem(parent);
	String valueName;
	if (parentNode->value().type() == 6) {
		valueName = parentNode->value().toString();
		__parents[valueName] = parent;
	} else {
		GVT_ASSERT(false, "Parent node value is not a string");
	}

	__tree[parent].push_back(node);
}

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

void Database::marshLeaf(unsigned char *buffer, DatabaseNode& leaf) {
		const char * name = leaf.name().c_str();
		memcpy(buffer, name, strlen(name) + 1);
		buffer += strlen(name) + 1;

		int type = leaf.value().type();
		memcpy(buffer, &type, sizeof(int));
		buffer += sizeof(int);

		//boost::variant<int, long, float, double, bool, unsigned long long, String, Uuid, glm::vec3> coreData;
		switch (type) {
		case 0: {
			int v = leaf.value().toInteger();
			memcpy(buffer, &(v), sizeof(int));
			buffer += sizeof(int);
			break;
		}
		case 1: {
			long v = leaf.value().toLong();
			memcpy(buffer, &(v), sizeof(long));
			buffer += sizeof(long);
			break;
		}
		case 2: {
			float v = leaf.value().toFloat();
			memcpy(buffer, &(v), sizeof(float));
			buffer += sizeof(float);
			break;
		}
		case 3: {
			double v = leaf.value().toDouble();
			memcpy(buffer, &(v), sizeof(double));
			buffer += sizeof(double);
			break;
		}
		case 4: {
			bool v = leaf.value().toBoolean();
			memcpy(buffer, &(v), sizeof(bool));
			buffer += sizeof(bool);
			break;
		}
		case 5: { //if pointer, handle according to what the pointer points
			if (strcmp(name,"bbox") == 0) {
				gvt::render::data::primitives::Box3D* bbox=
						(gvt::render::data::primitives::Box3D*)leaf.value().toULongLong();
				const float* v = glm::value_ptr(bbox->bounds_min);
				memcpy(buffer, v, sizeof(float) * 3);
				buffer += sizeof(float) * 3;

				v = glm::value_ptr(bbox->bounds_max);
				memcpy(buffer, v, sizeof(float) * 3);
				buffer += sizeof(float) * 3;
			} else if (strcmp(name,"ptr") == 0) { //if it is actually a pointer, invalidate - separate memory adresses
				memset(buffer, 0,sizeof(unsigned long long));
			} else
				GVT_ASSERT(false, "Pointer used in marsh");
			break;
		}
		case 6: {
			const char * vname = leaf.value().toString().c_str();
			memcpy(buffer, vname, strlen(vname) + 1);
			buffer += strlen(vname) + 1;
			break;
		}
		case 7: { // UUIDs are invalid across different trees
			GVT_ASSERT(false, "UUID used in marsh");
			break;
		}
		case 8: {
			const float* v = glm::value_ptr(leaf.value().tovec3());
			memcpy(buffer, v, sizeof(float) * 3);
			buffer += sizeof(float) * 3;
			break;
		}
		default:
			GVT_ASSERT(false, "Unknown variant type");
			break;
		}
	}

 DatabaseNode * Database::unmarshLeaf(unsigned char *buffer, Uuid parent) {

		String name =String((char*) buffer);
		buffer += strlen(name.c_str()) + 1;
		int type = *(int*) buffer;
		buffer += sizeof(int);

		Variant v;
		switch (type) {
		case 0: {
			v = *(int*) buffer;
			break;
		}
		case 1: {
			v = *(long*) buffer;
			break;
		}
		case 2: {
			v = *(float*) buffer;
			break;
		}
		case 3: {
			v = *(double*) buffer;
			break;
		}
		case 4: {
			v = *(bool*) buffer;
			break;
		}
		case 5: {
			if (name == String("bbox")) {
				float* fs = (float*) buffer;
				glm::vec3 bounds_min = glm::vec3(fs[0], fs[1], fs[2]);
				buffer += 3 * sizeof(float);

				fs = (float*) buffer;
				glm::vec3 bounds_max = glm::vec3(fs[0], fs[1], fs[2]);

				gvt::render::data::primitives::Box3D *meshbbox =
						new gvt::render::data::primitives::Box3D(bounds_min,
								bounds_max);

				v = (unsigned long long) meshbbox;
			} else if (name == String("ptr")) {
				v = (unsigned long long) NULL;
			} else
				GVT_ASSERT(false, "Pointer used in marsh");
			break;
		}
		case 6: {
			String s = String((char*) buffer);
			v = s;
			break;
		}
		case 7: {
			GVT_ASSERT(false, "UUID used in marsh");
			break;
		}
		case 8: {
			float* fs = (float*) buffer;
			v= glm::vec3(fs[0], fs[1], fs[2]);
			break;
		}
		default:
			GVT_ASSERT(false, "Unknown variant type");
			break;
		}

		return new DatabaseNode(name, v, Uuid(), parent);
	}

