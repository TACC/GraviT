#include "gvt/render/Context.h"

#include "gvt/core/Debug.h"

using gvt::core::DBNodeH;
using gvt::core::String;
using gvt::core::Uuid;
using namespace gvt::core::math;
using namespace gvt::render;

Context::Context()
: gvt::core::Context()
{}

Context::~Context()
{}

DBNodeH Context::createNodeFromType(String type, String name, Uuid parent)
{
    DBNodeH n = gvt::core::Context::createNode(type, name, parent);

    // TODO - make these for GraviT
    if (type == String("Camera"))
	{
		n += gvt::core::Context::createNode("focus");
		n += gvt::core::Context::createNode("eyePoint");
		n += gvt::core::Context::createNode("upVector");
		n += gvt::core::Context::createNode("cam2wrld");
	}
	else if (type == String("Film")) 
	{
		n += gvt::core::Context::createNode("width");
		n += gvt::core::Context::createNode("height");
	}
    else if (type == String("View"))
    {
        n += gvt::core::Context::createNode("width");
        n += gvt::core::Context::createNode("height");
        n += gvt::core::Context::createNode("viewAngle");
        n += gvt::core::Context::createNode("camera",Point4f());
        n += gvt::core::Context::createNode("focus",Point4f());
        n += gvt::core::Context::createNode("up",Vector4f());
        n += gvt::core::Context::createNode("parallelScale");
        n += gvt::core::Context::createNode("nearPlane");
        n += gvt::core::Context::createNode("farPlane");
    }
    else if (type == String("Attributes"))
    {
        n["Views"] += gvt::core::Context::createNodeFromType("View");
        n += gvt::core::Context::createNode("renderType","surface");
        n += gvt::core::Context::createNode("schedule","Image");
    }
    return n;
}

