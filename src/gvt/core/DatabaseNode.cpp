#include "gvt/core/DatabaseNode.h"
#include "gvt/core/Context.h"
#include "gvt/core/Debug.h"

using namespace gvt::core;

DatabaseNode* DatabaseNode::errNode = new DatabaseNode(String("error"), String("error"), Uuid(nil_uuid()), Uuid(nil_uuid()));

DatabaseNode::DatabaseNode()
{
}

DatabaseNode::DatabaseNode(String name, Variant value, Uuid uuid, Uuid parentUUID)
: p_uuid(uuid), p_name(name), p_value(value), p_parent(parentUUID)
{

}

//DatabaseNode& DatabaseNode::deRef()
//{
//    Context* ctx = Context::singleton();
//    Database& db = *(ctx->database());
////    DEBUG(value().toUuid().toString().toStdString());
//    DatabaseNode* ref = db.getItem(value().toUuid());
//    if (ref && (value().toUuid() != Uuid(0)))
//    {
////        DEBUG("success");
//        return *ref;
//    }
//    else
//    {
////        DEBUG("fail");
//        return *errNode;
//    }
//}

//DatabaseNode& DatabaseNode::operator[](const String& key)
//{

//    Context* ctx = Context::singleton();
//    Database& db = *(ctx->database());
//    DatabaseNode* child = db.getChildByName(UUID(), key);
//    if (!child)
////    else
//    {
//         child = &(ctx->createNode(key).getNode());
//         db.__tree[UUID()].push_back(child);
////         return *child;
////        std::cerr << "Key not found: " << key.toStdString() << std::endl;
////        return *errNode;
//    }
//    return *child;
//}

//DatabaseNode& DatabaseNode::operator+=(DatabaseNode& child)
//{
//    Context* ctx = Context::singleton();
//    Database& db = *(ctx->database());
//    child.setParentUUID(UUID());
////    db.addChild(UUID(), &child);
//            db.__tree[UUID()].push_back(&child);
//    child.propagateUpdate();
//    return *this;
//}

//DatabaseNode& DatabaseNode::operator=(const Variant val)
//{
////    DEBUG(p_name.toStdString());
//    setValue(val);
//    return *this;
//}

DatabaseNode::operator bool() const
{
//    if (this == errNode)
//        DEBUG("is errNode");
//    DEBUG("bool:");
//    DEBUG(p_uuid.toString().toStdString());
    return (p_uuid != Uuid(nil_uuid())) && (p_parent != Uuid(nil_uuid()));
}

Uuid DatabaseNode::UUID() 
{
    return p_uuid;
}

String DatabaseNode::name() 
{
    return p_name;
}

Uuid DatabaseNode::parentUUID() 
{
    return p_parent;
}

Variant DatabaseNode::value() 
{
    return p_value;
}

void DatabaseNode::setUUID(Uuid uuid) 
{
    p_uuid = uuid;
    //emit uuidChanged();
}

void DatabaseNode::setName(String name) 
{
    p_name = name;
    //emit nameChanged();
}

void DatabaseNode::setParentUUID(Uuid parentUUID) 
{
    p_parent = parentUUID;
    //emit parentUUIDChanged();
}

void DatabaseNode::setValue(Variant value) 
{
    p_value = value;
    //emit valueChanged();
    //propagateUpdate();
}

void DatabaseNode::propagateUpdate()
{
    DatabaseNode* pn;
    Context* ctx = Context::singleton();
    Database& db = *(ctx->database());
    pn = db.getItem(p_parent);
    Uuid cid = UUID();
    while (pn)
    {
        //emit pn->childChanged(cid);
        cid = pn->UUID();
        pn = db.getItem(pn->parentUUID());
    }
}

Vector<DatabaseNode*> DatabaseNode::getChildren() 
{
    Context* ctx = Context::singleton();
    Database& db = *(ctx->database());
    return db.getChildren(UUID());
}


/*******************

    DBNodeH

 *******************/


//DBNodeH DBNodeH::errNode();
//DBNodeH* DBNodeH::errNode = new DBNodeH(String("error"), Uuid(0), Uuid(0), 0);

DatabaseNode& DBNodeH::getNode()
{
    Context* ctx = Context::singleton();
    Database& db = *(ctx->database());
    DatabaseNode* n = db.getItem(_uuid);
    if (n)
        return *n;
    else
     return *DatabaseNode::errNode;
}

DBNodeH DBNodeH::deRef()
{
    Context* ctx = Context::singleton();
    Database& db = *(ctx->database());
//    DEBUG(value().toUuid().toString().toStdString());
    DatabaseNode& n = getNode();
    DatabaseNode* ref = db.getItem(variant_toUuid(n.value()));
    if (ref && (variant_toUuid(n.value()) != nil_uuid()))
    {
//        DEBUG("success");
        return DBNodeH(ref->UUID());
    }
    else
    {
        DEBUG_CERR("fail");
        return DBNodeH();
    }
}

DBNodeH DBNodeH::operator[](const String& key)
{
    Context* ctx = Context::singleton();
    Database& db = *(ctx->database());
    DatabaseNode* child = db.getChildByName(_uuid, key);
    if (!child)
//    else
    {
       child = &(ctx->createNode(key).getNode());
       db.getChildren(UUID()).push_back(child);
//         return *child;
//        std::cerr << "Key not found: " << key.toStdString() << std::endl;
//        return *errNode;
   }
   return DBNodeH(child->UUID());
}

DBNodeH& DBNodeH::operator+=(DBNodeH child)
{
    Context* ctx = Context::singleton();
    Database& db = *(ctx->database());
    child.setParentUUID(UUID());
//    db.addChild(UUID(), &child);
    db.getChildren(UUID()).push_back(&child.getNode());
    child.propagateUpdate();
    return *this;
}

DBNodeH& DBNodeH::operator=(Variant val)
{
//    DEBUG(p_name.toStdString());
    setValue(val);
    return *this;
}

bool DBNodeH::operator==(const Variant val)
{
   return value() == val;
}

DBNodeH::operator bool() const
{
//    if (this == errNode)
//        DEBUG("is errNode");
//    DEBUG("bool:");
//    DEBUG(p_uuid.toString().toStdString());
    return (_uuid != nil_uuid());
}

Uuid DBNodeH::UUID()
{
    DatabaseNode& n = getNode();
    return n.UUID();
}

String DBNodeH::name() 
{
    DatabaseNode& n = getNode();
    return n.name();
}

Uuid DBNodeH::parentUUID() 
{
    DatabaseNode& n = getNode();
    return n.parentUUID();
}

Variant DBNodeH::value() 
{
    DatabaseNode& n = getNode();
    return n.value();
}


void DBNodeH::setUUID(Uuid uuid) 
{
    _uuid=uuid;
    DatabaseNode& n = getNode();
    n.setUUID(uuid);
}

void DBNodeH::setName(String name) 
{
    DatabaseNode& n = getNode();
    n.setName(name);
}

void DBNodeH::setParentUUID(Uuid parentUUID)  
{
    DatabaseNode& n = getNode();
    n.setParentUUID(parentUUID);
}

void DBNodeH::setValue(Variant value) 
{
    DatabaseNode& n = getNode();
    n.setValue(value);
}

void DBNodeH::propagateUpdate()
{
    DatabaseNode& n = getNode();
    n.propagateUpdate();
}

void DBNodeH::connectValueChanged(const void * receiver, const char* method)
{
    DEBUG_CERR("gvt::core::DBNodeH::connectValueChanged not implemented");
    //receiver->connect(&getNode(),SIGNAL(valueChanged()), method);
}

void DBNodeH::connectChildChanged(const void * receiver,  const char* method)
{
    DEBUG_CERR("gvt::core::DBNodeH::connectChildChanged not implemented");
    //receiver->connect(&getNode(),SIGNAL(connectChildChanged()), method);
}

Vector<DBNodeH> DBNodeH::getChildren() 
{
    Context* ctx = Context::singleton();
    Database& db = *(ctx->database());
    Vector<DatabaseNode*> children = db.getChildren(UUID());
    Vector<DBNodeH> result;
    for(int i=0; i < children.size(); i++)
        result.push_back(DBNodeH(children[i]->UUID()));
    return result;
}
