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


   // <nodeName><int variant type><value>
   static void marshLeaf(unsigned char *buffer, DatabaseNode& leaf) {
		const char * name = leaf.p_name.c_str();
		memcpy(buffer, name, strlen(name) + 1);
		buffer += strlen(name) + 1;

		int type = leaf.p_value.type();
		memcpy(buffer, &type, sizeof(int));
		buffer += sizeof(int);

		//boost::variant<int, long, float, double, bool, unsigned long long, String, Uuid, glm::vec3> coreData;
		switch (type) {
		case 0: {
			int v = leaf.p_value.toInteger();
			memcpy(buffer, &(v), sizeof(int));
			buffer += sizeof(int);
			break;
		}
		case 1: {
			long v = leaf.p_value.toLong();
			memcpy(buffer, &(v), sizeof(long));
			buffer += sizeof(long);
			break;
		}
		case 2: {
			float v = leaf.p_value.toFloat();
			memcpy(buffer, &(v), sizeof(float));
			buffer += sizeof(float);
			break;
		}
		case 3: {
			double v = leaf.p_value.toDouble();
			memcpy(buffer, &(v), sizeof(double));
			buffer += sizeof(double);
			break;
		}
		case 4: {
			bool v = leaf.p_value.toBoolean();
			memcpy(buffer, &(v), sizeof(bool));
			buffer += sizeof(bool);
			break;
		}
		case 5: { //if pointer, handle according to what the pointer points
			if (strcmp(name,"bbox") == 0) {
				gvt::render::data::primitives::Box3D* bbox=
						(gvt::render::data::primitives::Box3D*)leaf.p_value.toULongLong();
				const float* v = glm::value_ptr(bbox->bounds_min);
				memcpy(buffer, v, sizeof(float) * 3);
				buffer += sizeof(float) * 3;

				v = glm::value_ptr(bbox->bounds_max);
				memcpy(buffer, v, sizeof(float) * 3);
				buffer += sizeof(float) * 3;
			} else if (strcmp(name,"ptr") == 0) { //if it is actually a pointer, invalidate - separate memory adresses
				memset(buffer, 0,sizeof(unsigned long long));
			} else
				GVT_ASSERT(true, "Pointer used in marsh");
			break;
		}
		case 6: {
			const char * vname = leaf.p_value.toString().c_str();
			memcpy(buffer, vname, strlen(vname) + 1);
			buffer += strlen(vname) + 1;
			break;
		}
		case 7: { // UUIDs are invalid across different trees
			GVT_ASSERT(true, "UUID used in marsh");
			break;
		}
		case 8: {
			const float* v = glm::value_ptr(leaf.p_value.tovec3());
			memcpy(buffer, &(v), sizeof(float) * 3);
			buffer += sizeof(float) * 3;
			break;
		}
		default:
			GVT_ASSERT(true, "Unknown variant type");
			break;
		}
	}

   //check marshLeaf
  static DatabaseNode * unmarshLeaf(unsigned char *buffer, Uuid parent) {

		String name =String((char*) buffer);
		buffer += strlen(name.c_str()) + 1;
		int type = *(int*) buffer;
		buffer += sizeof(int);

		Variant v;
		switch (type) {
		case 0: {
			v = (int*) buffer;
			break;
		}
		case 1: {
			v = (long*) buffer;
			break;
		}
		case 2: {
			v = (float*) buffer;
			break;
		}
		case 3: {
			v = (double*) buffer;
			break;
		}
		case 4: {
			v = (bool*) buffer;
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
				GVT_ASSERT(true, "Pointer used in marsh");
			break;
		}
		case 6: {
			String s = String((char*) buffer);
			v = s;
			break;
		}
		case 7: {
			GVT_ASSERT(true, "UUID used in marsh");
			break;
		}
		case 8: {
			float* fs = (float*) buffer;
			v= glm::vec3(fs[0], fs[1], fs[2]);
			break;
		}
		default:
			GVT_ASSERT(true, "Unknown variant type");
			break;
		}

		return new DatabaseNode(name, v, Uuid(), parent);
	}

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
