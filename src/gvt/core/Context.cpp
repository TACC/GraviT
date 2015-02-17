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
    DEBUG_CERR(String("createNode: ") + name + String(" ") + uuid_toString(np->UUID()));
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
    if (type == String("Session"))
    {
        n += createNode("Data");
        n += createNode("Views");
        n += createNode("Cameras");
        n += createNode("Scenes");
        n += createNode("Renderables");
        n += createNode("Reservoirs");
        n += createNode("PickingInfo");
        n += createNode("name");
        n += createNode("time");
        n += createNode("currentViewRef");
    }
    else if (type == String("DataSource"))
    {
        n += createNode("type");
        n += createNode("siloFile");
    }
    else if (type == String("View"))
    {
        n += createNode("type");
        n += createNode("title", String("Untitled"));
        //n += createNode("size", QSize(300,300));
        //n += createNode("position", QPoint(0,0));
    }
    else if (type == String("View3D"))
    {
        n += createNode("type", String("3D"));
        n += createNode("title", String("Untitled"));
        n += createNode("cameraRef");
        n += createNode("sceneRef");
        //n += createNode("size", QSize(300,300));
        //n += createNode("position", QPoint(0,0));
    }
    else if (type == String("Scene3D"))
    {
        n += createNode("Renderables");
        n += createNode("zExaggeration", 1.f);
    }
    else if (type == String("Renderable3DGrid"))
    {
        n += createNode("dataRef");
        n += createNode("dataVar");
        n += createNode("Operators");
        n += createNode("meshOpacity", 0.0f);
        //n += createNode("meshColor", QVector3D(1.0, 1.0, 1.0));
        n += createNode("propertyOpacity", 1.0f);
        n += createNode("logarithmic", false);
        n += createNode("colorMap", "Hot");
        n += createNode("zExaggeration", 1.0f);
        n += createNode("property", "NTG");
    }
    else if (type == String("Slice"))
    {
        n += createNode("enabled", false);
        n += createNode("axis");
        n += createNode("value");
    }
    else if (type == String("TripleSlice"))
    {
        n += createNode("enabled", false);
        n += createNode("i");
        n += createNode("j");
        n += createNode("k");
    }
    else if (type == String("PerspectiveCamera"))
    {
        //n += createNode("rotation", QVector3D(0,0,0));
        n += createNode("zoom", 0.f);
    }
    else if (type == String("Threshold"))
    {
        n += createNode("enabled", true);
        n += createNode("min", 0.f);
        n += createNode("max", 1.f);
    }
    return n;
}

