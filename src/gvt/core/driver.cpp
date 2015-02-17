#include "gvt/core/Context.h"
#include "gvt/core/Database.h"
#include "gvt/core/DatabaseNode.h"
#include "gvt/core/Debug.h"

using namespace gvt::core;

int main(int argc, char** argv)
{
	Context& ctx = *Context::singleton();
	Database& db = *ctx.database();

	DEBUG_CERR(String("Created database with root node uuid ") + uuid_toString(ctx.getRootNode().UUID()));

	DBNodeH session = ctx.createNodeFromType(String("Session"),ctx.getRootNode().UUID());
	DBNodeH renderable = ctx.createNodeFromType(String("Renderable3DGrid"),ctx.getRootNode().UUID());

	DEBUG_CERR(String("Created session with node uuid ") + uuid_toString(session.UUID()));
	DEBUG_CERR(String("Created renderable with node uuid ") + uuid_toString(renderable.UUID()));

	session["foo"] = String("foo");
	renderable["bar"] = 5;

	db.printTree(ctx.getRootNode().UUID());

	return 0;
}