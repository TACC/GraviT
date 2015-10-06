#ifndef GVT_CORE_CONTEXT_H
#define GVT_CORE_CONTEXT_H

#include <gvt/core/Database.h>
#include "gvt/core/DatabaseNode.h"
#include "gvt/core/Types.h"

namespace gvt {
namespace core {
  /// context base class for GraviT internal state
  /** base class for GraviT internal state. 
  The context contains the object-store database and helper methods to create and manage the internal state.
  */
class CoreContext {
public:
  virtual ~CoreContext();

  /// return the context singleton
  static CoreContext *instance();

  /// return the object-store database
  Database *database() {
    GVT_ASSERT(__database != nullptr,
               "The context seems to be uninitialized.");
    return __database;
  }

  /// return a handle to the root of the object-store hierarchy
  DBNodeH getRootNode() {
    GVT_ASSERT(__database != nullptr,
               "The context seems to be uninitialized.");
    return __rootNode;
  }

  /// return a handle to the node with matching UUID
  DBNodeH getNode(Uuid);

  /// create a node in the database
  /** \param name the node name
      \param val the node value
      \param parent the uuid of the parent node
      */
  DBNodeH createNode(String name, Variant val = Variant(String("")),
                     Uuid parent = nil_uuid());

  /// create a node of the specified type in the database 
  DBNodeH createNodeFromType(String);

  /// create a node of the specified type with the specifed parent uuid
  DBNodeH createNodeFromType(String, Uuid);

  /// create a node of the specified type with the specified name and parent
  /** \param type the type of node to create
      \param name the node name
      \param parent the uuid of the parent node
      */
  DBNodeH createNodeFromType(String type, String name,
                                     Uuid parent = nil_uuid());
protected:
  CoreContext();
  static CoreContext *__singleton;
  Database *__database = nullptr;
  DBNodeH __rootNode;
};
}
}

#endif // GVT_CORE_CONTEXT_H
