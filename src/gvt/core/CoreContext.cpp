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

void CoreContext::marshNode(unsigned char *buffer, DBNodeH& node) {

	const char * parentName =
			__database->getItem(node.parentUUID())->value().toString().c_str();

	memcpy(buffer, parentName, strlen(parentName) + 1);
	buffer += strlen(parentName) + 1;

	Database::marshLeaf(buffer, node.getNode());
	buffer += CONTEXT_LEAF_MARSH_SIZE;

	Vector<DBNodeH> children = node.getChildren();
	int nChildren = children.size();
	memcpy(buffer, &(nChildren), sizeof(int));
	buffer += sizeof(int);
	for (auto leaf : children) {
		if (leaf.getChildren().size() > 0) GVT_ASSERT(false, "leaf with children");
		Database::marshLeaf(buffer, leaf.getNode());
		buffer += CONTEXT_LEAF_MARSH_SIZE;
	}
}

// check marsh for buffer structure
  DBNodeH CoreContext::unmarsh(unsigned char *buffer){

	  String grandParentName = std::string((char*) buffer);
	  buffer+=grandParentName.size()+1;

	  //searchs where the node is going to be added/updated
	  DBNodeH grandParentNode;
	  if (__database->hasParentNode(grandParentName)){
		  grandParentNode = DBNodeH(getNode(__database->getParentNode(grandParentName)->UUID()));
	  } else {
		GVT_ASSERT(false,"Grand parent node not found");
	  }


	  DatabaseNode * unmarshedParent = Database::unmarshLeaf(buffer, grandParentNode.UUID());
	  DBNodeH unmarshedParentHandler = DBNodeH(unmarshedParent->UUID());


	  //checks if non-leaf parent node exists
	  if (unmarshedParent->value().type() == 6) {

			//if does not exist, add new node
		  	  String unmarshedParentValueName = unmarshedParent->value().toString();
			if (!__database->hasParentNode(
					unmarshedParentValueName)) {

				__database->setItem(unmarshedParent);

				buffer += CONTEXT_LEAF_MARSH_SIZE;

				int nChildren = *(int*) buffer;
				buffer += sizeof(int);

				for (int i = 0; i < nChildren; i++) {
					DatabaseNode *unmarshedChild = Database::unmarshLeaf(
							buffer, unmarshedParentHandler.UUID());
					DBNodeH unmarshedChildHandler = DBNodeH(
							unmarshedChild->UUID());
					__database->setItem(unmarshedChild);
					buffer += CONTEXT_LEAF_MARSH_SIZE;

				}

			//if exists udpate children data
			} else {

				DBNodeH existingParentNode = DBNodeH(getNode(__database->getParentNode(
						unmarshedParent->value().toString())->UUID()));


				buffer += CONTEXT_LEAF_MARSH_SIZE;

				int nChildren = *(int*) buffer;
				buffer += sizeof(int);

				if (existingParentNode.getChildren().size() != nChildren){
					GVT_ASSERT(false, "Updating node with different number of children");
				}

				for (int i = 0; i < nChildren; i++) {
					DatabaseNode *unmarshedChild = Database::unmarshLeaf(
							buffer, existingParentNode.UUID());
					DBNodeH unmarshedChildHandler = DBNodeH(
							unmarshedChild->UUID());

					existingParentNode[unmarshedChild->name()] = unmarshedChild->value();

					buffer += CONTEXT_LEAF_MARSH_SIZE;
				}

				unmarshedParentHandler = existingParentNode;

			}
		}
	  else {
		  GVT_ASSERT(false, "parent node is unexpectadly a non-string");
	  }

	  return unmarshedParentHandler;

  }


void CoreContext::syncContext(){

	const int rankSize = MPI::COMM_WORLD.Get_size();
	const int myRank = MPI::COMM_WORLD.Get_rank();
	int bytesRequired=0;

	std::vector<int> mySyncTable(rankSize, 0);
	std::vector<int> syncTable(rankSize,0);
	std::vector<unsigned char> buf;

	mySyncTable[myRank]=__nodesToSync.size();
	MPI::COMM_WORLD.Allreduce(&mySyncTable[0], &syncTable[0], rankSize, MPI_INT, MPI_MAX);

	for (int i = 0; i < rankSize; i++) {
		if (syncTable[i] == 0)
			continue;
		for (int message = 0; message < syncTable[i]; message++) {

			if (i == myRank) {
				DBNodeH node = __nodesToSync[message];
				bytesRequired = CONTEXT_LEAF_MARSH_SIZE
						* (node.getChildren().size() + 1);
				buf.reserve(bytesRequired);
				marshNode(&buf[0], node);
			}

			MPI::COMM_WORLD.Bcast(&bytesRequired, 1, MPI::INT, i);
			buf.reserve(bytesRequired);
			MPI::COMM_WORLD.Bcast(&buf[0], bytesRequired, MPI_UNSIGNED_CHAR, i);

			if (i != myRank)  {
				DBNodeH node = unmarsh(&buf[0]);
				//database()->printTree(node.UUID(), 4, std::cout);

			}

		}
	}

	MPI::COMM_WORLD.Barrier();

  }
