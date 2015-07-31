#ifndef GVT_CORE_CONTEXT_H
#define GVT_CORE_CONTEXT_H

#include <gvt/core/Database.h>
#include "gvt/core/DatabaseNode.h"
#include "gvt/core/Types.h"

namespace gvt {
namespace core {
class CoreContext {
public:
  virtual ~CoreContext();

  static CoreContext *instance();
  Database *database() {
    GVT_ASSERT(__database != nullptr,
               "The context seems to be uninitialized.");
    return __database;
  }

  DBNodeH getRootNode() {
    GVT_ASSERT(__database != nullptr,
               "The context seems to be uninitialized.");
    return __rootNode;
  }
  DBNodeH getNode(Uuid);

  DBNodeH createNode(String name, Variant val = Variant(String("")),
                     Uuid parent = nil_uuid());
  DBNodeH createNodeFromType(String);
  DBNodeH createNodeFromType(String, Uuid);
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
