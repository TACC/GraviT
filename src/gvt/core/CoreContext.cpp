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
  std::vector<MarshedDatabaseNode> buf;
  buf.reserve(1);
  DatabaseNode *root;

  if (MPI::COMM_WORLD.Get_rank() == 0) {

		root = new DatabaseNode(String("GraviT"),
				String("GVT ROOT"), Uuid(), Uuid::null());

		marsh(buf, *root);
	}

	MPI::COMM_WORLD.Bcast(&buf[0], CONTEXT_LEAF_MARSH_SIZE, MPI_UNSIGNED_CHAR,
			0);

	if (MPI::COMM_WORLD.Get_rank() != 0) {
		root = unmarsh(buf,1);
	}

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
  DBNodeH node = DBNodeH(np->UUID());
  __database->setItem(np);
  GVT_DEBUG(DBG_LOW, "createNode: " << name << " " << np->UUID().toString());
  return node;
}

DBNodeH CoreContext::createNodeFromType(String type, Uuid parent) { return createNodeFromType(type, type, parent); }

DBNodeH CoreContext::createNodeFromType(String type) { return createNodeFromType(type, type); }

DBNodeH CoreContext::createNodeFromType(String type, String name, Uuid parent) {
  DBNodeH n = createNode(type, name, parent);

  // TODO - make these for GraviT

  return n;
}

void CoreContext::marsh(std::vector<MarshedDatabaseNode>& messagesBuffer, DatabaseNode& node) {


	messagesBuffer.push_back(MarshedDatabaseNode());
	unsigned char* buffer = &(messagesBuffer.back().data[0]);

	Uuid parentUUID = node.parentUUID();
	memcpy(buffer, &parentUUID, sizeof(Uuid));
	buffer += sizeof(Uuid);


	Vector<DatabaseNode*> children;

	//if syncing root node
	if (CoreContext::__singleton != nullptr ){
		children = node.getChildren();
	}

	int nChildren = children.size();

	__database->marshLeaf(buffer, node);

	for (int i = 0; i < nChildren; i++) {
		DatabaseNode* leaf = children[i];
		marsh(messagesBuffer, *leaf);
	}
}

DatabaseNode* CoreContext::unmarsh(
		std::vector<MarshedDatabaseNode>& messagesBuffer, int nNodes) {

	DatabaseNode *unmarshedParent;

	for (int i = 0; i < nNodes; i++) {

		unsigned char* buffer = &(messagesBuffer[i].data[0]);

		Uuid grandParentUUID;
		memcpy(&grandParentUUID, buffer, sizeof(Uuid));
		buffer += sizeof(Uuid);

		DatabaseNode *grandParentNode;

		// searchs where the node is going to be added/updated
		if (__database->hasNode(grandParentUUID)) {
			grandParentNode = __database->getItem(grandParentUUID);
		} else { // should only be the case of syncing root
			unmarshedParent = __database->unmarshLeaf(buffer, Uuid::null());
			return unmarshedParent;
		}

		unmarshedParent = __database->unmarshLeaf(buffer, grandParentUUID);

		// if does not exist add node
		if (!__database->hasNode(unmarshedParent->UUID())) {

			__database->setItem(unmarshedParent);

			// if exists udpate data
		} else {
			if (unmarshedParent->name() != String("ptr"))
			__database->getItem(unmarshedParent->UUID())->setValue(
					unmarshedParent->value());
		}
	}

	return unmarshedParent;
}

void CoreContext::syncContext(){

  const int rankSize = MPI::COMM_WORLD.Get_size();
  const int myRank = MPI::COMM_WORLD.Get_rank();
  int nTreeNodeEntries = 0;

  std::vector<int> mySyncTable(rankSize, 0); //array with the # tree-nodes to send
  std::vector<int> syncTable(rankSize, 0); //reduced array with the # tree-nodes to send from all nodes
  std::vector<MarshedDatabaseNode> buf;

  mySyncTable[myRank] = __nodesToSync.size();
  MPI::COMM_WORLD.Allreduce(&mySyncTable[0], &syncTable[0], rankSize, MPI_INT, MPI_MAX);

  for (int i = 0; i < rankSize; i++) {
    if (syncTable[i] == 0) continue;
    for (int treeNode = 0; treeNode < syncTable[i]; treeNode++) {

      if (i == myRank) {
        DatabaseNode &node = __nodesToSync[treeNode].getNode();
        marsh(buf, node);
        nTreeNodeEntries = buf.size();
      }

      //agree on message length, strings length and # leaf are arbitrary
      MPI::COMM_WORLD.Bcast(&nTreeNodeEntries, 1, MPI::INT, i);
      buf.reserve(nTreeNodeEntries);
      MPI::COMM_WORLD.Bcast(&buf[0], nTreeNodeEntries*CONTEXT_LEAF_MARSH_SIZE, MPI_UNSIGNED_CHAR, i);

      if (i != myRank) {
        unmarsh(buf,nTreeNodeEntries);
      }

     buf.clear();
    }
  }

  MPI::COMM_WORLD.Barrier();

  __nodesToSync.clear();
}
