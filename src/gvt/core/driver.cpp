#include "gvt/core/Context.h"
#include "gvt/core/Database.h"
#include "gvt/core/DatabaseNode.h"
#include "gvt/core/Debug.h"

using namespace gvt::core;

int main(int argc, char** argv)
{
	Context& ctx = *Context::singleton();
	Database& db = *ctx.database();

	GVT_DEBUG(DBG_ALWAYS,"Created database with root node uuid " << uuid_toString(ctx.getRootNode().UUID()));

	DBNodeH session = ctx.createNodeFromType(String("Session"),ctx.getRootNode().UUID());
	DBNodeH renderable = ctx.createNodeFromType(String("Renderable3DGrid"),ctx.getRootNode().UUID());

	GVT_DEBUG(DBG_ALWAYS,"Created session with node uuid " << uuid_toString(session.UUID()));
	GVT_DEBUG(DBG_ALWAYS,"Created renderable with node uuid " << uuid_toString(renderable.UUID()));

	session["foo"] = String("foo");
	renderable["bar"] = 5;

	db.printTree(ctx.getRootNode().UUID());

	return 0;
}