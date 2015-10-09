#ifndef GVT_RENDER_CONTEXT_H
#define GVT_RENDER_CONTEXT_H

#include <gvt/core/CoreContext.h>

namespace gvt {
	namespace render {
		/// internal context for GraviT ray tracing renderer
		/** \sa CoreContext
		*/
		class RenderContext : public gvt::core::CoreContext
		{
		public:
			static void CreateContext();
			virtual ~RenderContext(); 
			gvt::core::DBNodeH createNodeFromType(gvt::core::String type, 
														  gvt::core::String name, 
														  gvt::core::Uuid parent = gvt::core::nil_uuid());
			static RenderContext* instance();

		protected:
			RenderContext();

		};
	}
}

#endif // GVT_RENDER_CONTEXT_H
