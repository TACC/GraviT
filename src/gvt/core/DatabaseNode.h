#ifndef GVT_CORE_DATABASE_NODE_H
#define GVT_CORE_DATABASE_NODE_H

#include "gvt/core/Types.h"

namespace gvt {
    namespace core {
        class DatabaseNode
        {


            Uuid        p_uuid;
            String      p_name;
            Uuid        p_parent;
            Variant     p_value;

        public:
            DatabaseNode(String name, Variant value, Uuid uuid, Uuid parentUUID);

            Uuid       UUID();
            String     name();
            Uuid       parentUUID();
            Variant    value();

            void setUUID(Uuid uuid);
            void setName(String name);
            void setParentUUID(Uuid parentUUID);
            void setValue(Variant value);

            Vector<DatabaseNode*> getChildren();

            void propagateUpdate();
            explicit operator bool() const;

            static DatabaseNode* errNode;

        };

        class DBNodeH
        {
        public:
            explicit DBNodeH(Uuid uuid = Uuid(nil_uuid())) { _uuid = uuid; }
            Uuid       UUID();
            String     name();
            Uuid       parentUUID();
            Variant    value();

            void setUUID(Uuid uuid);
            void setName(String name);
            void setParentUUID(Uuid parentUUID);
            void setValue(Variant value);

            Vector<DBNodeH> getChildren();

            void propagateUpdate();
            DBNodeH deRef();

            DatabaseNode& getNode();
            
            DBNodeH operator[](const String& key);
            DBNodeH& operator+=(DBNodeH child);
            DBNodeH& operator=(Variant val);
            bool operator==(const Variant val);
            explicit operator bool() const;

            void connectValueChanged(const void * receiver, const char* method);
            void connectChildChanged(const void * receiver, const char* method);

private:
            Uuid        _uuid;
         };
    }
}

#endif // GVT_CORE_DATABASE_NODE_H
