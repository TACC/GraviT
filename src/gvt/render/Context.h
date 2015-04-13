#ifndef GVT_RENDER_CONTEXT_H
#define GVT_RENDER_CONTEXT_H

#include <gvt/core/Context.h>

namespace gvt {
	namespace render {
		class Context : public gvt::core::Context
		{
		public:
			Context();
			virtual ~Context(); 

			static Context* singleton();

			virtual DBNodeH createNodeFromType(String type, String name, Uuid parent = nil_uuid());
		};
	}
}

#endif // GVT_RENDER_CONTEXT_H