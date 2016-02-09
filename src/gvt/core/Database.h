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
#ifndef GVT_CORE_DATABASE_H
#define GVT_CORE_DATABASE_H

#include "gvt/core/DatabaseNode.h"
#include "gvt/core/Types.h"

#include <iostream>

namespace gvt {
namespace core {

typedef Vector<DatabaseNode *> ChildList;
/// object-store database for GraviT
/**
object store database for GraviT. The stored objects are contained in DatabaseNode objects.
The database and the objects are typicallly accessed using the CoreContext singleton, which
returns DBNodeH handles to the database objects.
*/
class Database {
public:
  Database();
  ~Database();

  /// does the database contain a node with the given unique id
  bool hasNode(Uuid);
  // does the database contain this node
  bool hasNode(DatabaseNode *);

  // get the database node with the given unique id
  DatabaseNode *getItem(Uuid);
  /// add this node to the database
  void setItem(DatabaseNode *);
  /// set this node as the root of the database hierarchy
  void setRoot(DatabaseNode *);
  /// remove the node with the given unique id
  void removeItem(Uuid);
  /// add the given node as a child of the node with the given uuid
  void addChild(Uuid, DatabaseNode *);
  /// return the value of the child node with the given name
  /// that is a child of the parent with the given uuid
  Variant getChildValue(Uuid, String);
  /// set the value of the child node with the given name
  /// that is a child of the parent with the given uuid
  void setChildValue(Uuid, String, Variant);
  /// return the children node pointers of the parent node with the given uuid
  ChildList &getChildren(Uuid);
  /// return the child node with the given name
  /// that is a child of the parent with the given uuid
  DatabaseNode *getChildByName(Uuid, String);

  /// return the value of the node with the given uuid
  Variant getValue(Uuid);
  /// set the value of the node with the given uuid
  void setValue(Uuid, Variant);
  /// print the given node and its immediate children
  void print(const Uuid &parent, const int depth = 0, std::ostream &os = std::cout);
  /// print the complete database hierarchy rooted at the given node
  void printTree(const Uuid &parent, const int depth = 0, std::ostream &os = std::cout);

private:
  Map<Uuid, DatabaseNode *> __nodes;
  Map<Uuid, ChildList> __tree;
};
}
}
#endif // GVT_CORE_DATABASE_H
