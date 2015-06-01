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

			virtual gvt::core::DBNodeH createNodeFromType(gvt::core::String type, 
														  gvt::core::String name, 
														  gvt::core::Uuid parent = gvt::core::nil_uuid()
														  );
		};
	}
}

#endif // GVT_RENDER_CONTEXT_H