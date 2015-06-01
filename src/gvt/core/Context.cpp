#include "gvt/core/Context.h"
#include "gvt/core/Debug.h"

using namespace gvt::core;

Context* Context::__singleton = NULL;

Context::Context()
{
    __database = new Database();
    DatabaseNode* root = new DatabaseNode(String("GraviT"),String("GVT ROOT"),make_uuid(),nil_uuid());
    __database->setRoot(root);
    __rootNode = DBNodeH(root->UUID());
}

Context::~Context()
{
    delete __database;
}

Context* Context::singleton()
{
	if (!__singleton) { __singleton = new Context(); }
	return __singleton;
}

DBNodeH Context::getNode(Uuid node)
{
	DatabaseNode* n = __database->getItem(node);
	if (n) return DBNodeH(n->UUID());
	else return DBNodeH();
}

DBNodeH Context::createNode(String name, Variant val, Uuid parent)
{
    DatabaseNode* np = new DatabaseNode(name, val, make_uuid(), parent);
    __database->setItem(np);
    GVT_DEBUG(DBG_LOW,"createNode: " << name << " " << uuid_toString(np->UUID()));
    return DBNodeH(np->UUID());
}

DBNodeH Context::createNodeFromType(String type, Uuid parent)
{
    return createNodeFromType(type, type, parent);
}

DBNodeH Context::createNodeFromType(String type)
{
    return createNodeFromType(type, type);
}

DBNodeH Context::createNodeFromType(String type, String name, Uuid parent)
{
    DBNodeH n = createNode(type, name, parent);

    // TODO - make these for GraviT
 
    return n;
}

