#ifndef GVT_CORE_DATABASE_H
#define GVT_CORE_DATABASE_H

#include "gvt/core/DatabaseNode.h"
#include "gvt/core/Types.h"

#include <iostream>

namespace gvt {
    namespace core {

        typedef Vector<DatabaseNode*> ChildList;
        /// object-store database for GraviT
        /**
        object store database for GraviT. The stored objects are contained in DatabaseNode objects.
        The database and the objects are typicallly accessed using the CoreContext singleton, which
        returns DBNodeH handles to the database objects.
        */
        class Database
        {
        public:
            Database();
            ~Database();

            /// does the database contain a node with the given unique id
            bool hasNode(Uuid);
            // does the database contain this node
            bool hasNode(DatabaseNode*);

            // get the database node with the given unique id
            DatabaseNode* getItem(Uuid);
            /// add this node to the database
            void setItem(DatabaseNode*);
            /// set this node as the root of the database hierarchy
            void setRoot(DatabaseNode*);
            /// remove the node with the given unique id
            void removeItem(Uuid);
            /// add the given node as a child of the node with the given uuid
            void addChild(Uuid, DatabaseNode*);
            /// return the value of the child node with the given name 
            /// that is a child of the parent with the given uuid
            Variant getChildValue(Uuid, String);
            /// set the value of the child node with the given name
            /// that is a child of the parent with the given uuid
            void setChildValue(Uuid, String, Variant);
            /// return the children node pointers of the parent node with the given uuid
            ChildList& getChildren(Uuid);
            /// return the child node with the given name
            /// that is a child of the parent with the given uuid
            DatabaseNode* getChildByName(Uuid, String);

            /// return the value of the node with the given uuid
            Variant getValue(Uuid);
            /// set the value of the node with the given uuid
            void setValue(Uuid, Variant);
            /// print the given node and its immediate children
            void print(const Uuid& parent, const int depth =0, std::ostream& os=std::cout);
            /// print the complete database hierarchy rooted at the given node
            void printTree(const Uuid& parent, const int depth=0, std::ostream& os=std::cout);

        private:
            Map<Uuid,DatabaseNode*>     __nodes;
            Map<Uuid, ChildList>        __tree;
        };
    }
}
#endif // GVT_CORE_DATABASE_H
