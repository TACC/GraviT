#ifndef GVT_CORE_CONTEXT_H
#define GVT_CORE_CONTEXT_H

#include "gvt/core/Database.h"
#include "gvt/core/DatabaseNode.h"
#include "gvt/core/Types.h"

namespace gvt {
	namespace core {
		class Context 
		{
		public:
			Context();
			~Context(); 

			static Context* singleton();
			Database* database() { return __database; }

			DBNodeH getRootNode() { return __rootNode; }
			DBNodeH getNode(Uuid node);

			DBNodeH createNodeFromType(String type);
			DBNodeH createNode(String name, Variant val = Variant(String("")));

		private:
			static Context* 	__singleton;
			Database*			__database;
			DBNodeH 			__rootNode;
		};
	}
}

#endif // GVT_CORE_CONTEXT_H