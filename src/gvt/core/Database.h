#ifndef GVT_CORE_DATABASE_H
#define GVT_CORE_DATABASE_H

#include "gvt/core/DatabaseNode.h"
#include "gvt/core/Types.h"

#include <iostream>

namespace gvt {
    namespace core {

        typedef Vector<DatabaseNode*> ChildList;

        class Database
        {
        public:
            Database();

            bool hasNode(Uuid);
            bool hasNode(DatabaseNode*);

            DatabaseNode* getItem(Uuid);
            void setItem(DatabaseNode*);
            void removeItem(Uuid);

            void addChild(Uuid, DatabaseNode*);
            Variant getChildValue(Uuid, String);
            void setChildValue(Uuid, String, Variant);
            ChildList& getChildren(Uuid);
            DatabaseNode* getChildByName(Uuid, String);

            Variant getValue(Uuid);
            void setValue(Uuid, Variant);

            void print(const Uuid& parent, const int depth =0, std::ostream& os=std::cout);
            void printtree(const Uuid& parent, const int depth=0, std::ostream& os=std::cout);

        private:
            Map<Uuid,DatabaseNode*>     __nodes;
            Map<Uuid, ChildList>        __tree;
        };
    }
}
#endif // GVT_CORE_DATABASE_H
