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
			DBNodeH getNode(Uuid);

			DBNodeH createNodeFromType(String);
			DBNodeH createNodeFromType(String, Uuid);
			DBNodeH createNodeFromType(String type, String name, Uuid parent = nil_uuid());
			DBNodeH createNode(String name, Variant val = Variant(String("")), Uuid parent = nil_uuid());

		private:
			static Context* 	__singleton;
			Database*			__database;
			DBNodeH 			__rootNode;
		};
	}
}

#endif // GVT_CORE_CONTEXT_H