#include "gvt/render/RenderContext.h"

#include "gvt/core/Debug.h"

using gvt::core::DBNodeH;
using gvt::core::String;
using gvt::core::Uuid;
using namespace gvt::core::math;
using namespace gvt::render;

RenderContext::RenderContext()
: gvt::core::CoreContext()
{}

void RenderContext::CreateContext()
{
	if(!__singleton) {
		__singleton = new RenderContext();
	}
}

RenderContext* RenderContext::instance() {
    // if (__singleton != nullptr) {
    //     __singleton = new RenderContext();
    // }
    return static_cast<RenderContext*>(CoreContext::instance());
}

RenderContext::~RenderContext()
{}

DBNodeH RenderContext::createNodeFromType(String type, String name, Uuid parent)
{

    DBNodeH n = gvt::core::CoreContext::createNode(type, name, parent);

    // TODO - make these for GraviT
    if (type == String("Camera"))
	{
		n += gvt::core::CoreContext::createNode("focus");
		n += gvt::core::CoreContext::createNode("eyePoint");
		n += gvt::core::CoreContext::createNode("upVector");
		n += gvt::core::CoreContext::createNode("cam2wrld");
	}
	else if (type == String("Film")) 
	{
		n += gvt::core::CoreContext::createNode("width");
		n += gvt::core::CoreContext::createNode("height");
	}
    else if (type == String("View"))
    {
        n += gvt::core::CoreContext::createNode("width");
        n += gvt::core::CoreContext::createNode("height");
        n += gvt::core::CoreContext::createNode("viewAngle");
        n += gvt::core::CoreContext::createNode("camera",Point4f());
        n += gvt::core::CoreContext::createNode("focus",Point4f());
        n += gvt::core::CoreContext::createNode("up",Vector4f());
        n += gvt::core::CoreContext::createNode("parallelScale");
        n += gvt::core::CoreContext::createNode("nearPlane");
        n += gvt::core::CoreContext::createNode("farPlane");
    }
    else if (type == String("Dataset"))
	{
		n += gvt::core::CoreContext::createNode("schedule");
		n += gvt::core::CoreContext::createNode("render_type");
		n += gvt::core::CoreContext::createNode("topology");
		n += gvt::core::CoreContext::createNode("Dataset_Pointer");
		n += gvt::core::CoreContext::createNode("accel_type");
		n += gvt::core::CoreContext::createNode("Mesh_Pointer");
	}
    else if (type == String("Attributes"))
    {
        n["Views"] += gvt::core::CoreContext::createNodeFromType("View");
        n += gvt::core::CoreContext::createNode("renderType","surface");
        n += gvt::core::CoreContext::createNode("schedule","Image");
    }
    return n;
}

