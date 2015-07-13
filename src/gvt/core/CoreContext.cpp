#include "gvt/core/CoreContext.h"
#include "gvt/core/Debug.h"

using namespace gvt::core;

CoreContext* CoreContext::__singleton = NULL;

CoreContext::CoreContext()
{
    __database = new Database();
    DatabaseNode* root = new DatabaseNode(String("GraviT"),String("GVT ROOT"),make_uuid(),nil_uuid());
    __database->setRoot(root);
    __rootNode = DBNodeH(root->UUID());
}

CoreContext::~CoreContext()
{
    delete __database;
}

CoreContext* CoreContext::instance()
{
	if (__singleton != NULL) {return __singleton;}
}

DBNodeH CoreContext::getNode(Uuid node)
{
	DatabaseNode* n = __database->getItem(node);
	if (n) return DBNodeH(n->UUID());
	else return DBNodeH();
}

DBNodeH CoreContext::createNode(String name, Variant val, Uuid parent)
{
    DatabaseNode* np = new DatabaseNode(name, val, make_uuid(), parent);
    __database->setItem(np);
    GVT_DEBUG(DBG_LOW,"createNode: " << name << " " << uuid_toString(np->UUID()));
    return DBNodeH(np->UUID());
}

DBNodeH CoreContext::createNodeFromType(String type, Uuid parent)
{
    return createNodeFromType(type, type, parent);
}

DBNodeH CoreContext::createNodeFromType(String type)
{
    return createNodeFromType(type, type);
}

DBNodeH CoreContext::createNodeFromType(String type, String name, Uuid parent)
{
    DBNodeH n = createNode(type, name, parent);

    // TODO - make these for GraviT
 
    return n;
}

