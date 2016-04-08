/* =======================================================================================
   This file is released as part of GraviT - scalable, platform independent ray tracing
   tacc.github.io/GraviT

   Copyright 2013-2015 Texas Advanced Computing Center, The University of Texas at Austin
   All rights reserved.

   Licensed under the BSD 3-Clause License, (the "License"); you may not use this file
   except in compliance with the License.
   A copy of the License is included with this software in the file LICENSE.
   If your copy does not contain the License, you may obtain a copy of the License at:

       http://opensource.org/licenses/BSD-3-Clause

   Unless required by applicable law or agreed to in writing, software distributed under
   the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
   KIND, either express or implied.
   See the License for the specific language governing permissions and limitations under
   limitations under the License.

   GraviT is funded in part by the US National Science Foundation under awards ACI-1339863,
   ACI-1339881 and ACI-1339840
   ======================================================================================= */
#include "gvt/core/DatabaseNode.h"
#include "gvt/core/CoreContext.h"
#include "gvt/core/Debug.h"

using namespace gvt::core;

DatabaseNode *DatabaseNode::errNode = new DatabaseNode(String("error"), String("error"), Uuid::null(), Uuid::null());

DatabaseNode::DatabaseNode(String name, Variant value, Uuid uuid, Uuid parentUUID)
    : p_uuid(uuid), p_name(name), p_value(value), p_parent(parentUUID) {}

DatabaseNode::operator bool() const { return (!p_uuid.isNull() && !p_parent.isNull()); }

Uuid DatabaseNode::UUID() { return p_uuid; }

String DatabaseNode::name() { return p_name; }

Uuid DatabaseNode::parentUUID() { return p_parent; }

Variant DatabaseNode::value() { return p_value; }

void DatabaseNode::setUUID(Uuid uuid) { p_uuid = uuid; }

void DatabaseNode::setName(String name) { p_name = name; }

void DatabaseNode::setParentUUID(Uuid parentUUID) { p_parent = parentUUID; }

void DatabaseNode::setValue(Variant value) { p_value = value; }

void DatabaseNode::propagateUpdate() {
  DatabaseNode *pn;
  CoreContext *ctx = CoreContext::instance();
  Database &db = *(ctx->database());
  pn = db.getItem(p_parent);
  Uuid cid = UUID();
  while (pn) {
    cid = pn->UUID();
    pn = db.getItem(pn->parentUUID());
  }
}

Vector<DatabaseNode *> DatabaseNode::getChildren() {
  CoreContext *ctx = CoreContext::instance();
  Database &db = *(ctx->database());
  return db.getChildren(UUID());
}

/*******************

    DBNodeH

 *******************/

DBNodeH::DBNodeH(Uuid uuid) : _uuid(uuid) {}

DatabaseNode &DBNodeH::getNode() {
  CoreContext *ctx = CoreContext::instance();
  Database &db = *(ctx->database());
  DatabaseNode *n = db.getItem(_uuid);
  if (n)
    return *n;
  else
    return *DatabaseNode::errNode;
}

DBNodeH DBNodeH::deRef() {
  CoreContext *ctx = CoreContext::instance();
  Database &db = *(ctx->database());
  DatabaseNode &n = getNode();
  DatabaseNode *ref = db.getItem(n.value().toUuid());
  if (ref && !n.value().toUuid().isNull()) {
    return DBNodeH(ref->UUID());
  } else {
    GVT_DEBUG(DBG_SEVERE, "DBNodeH deRef failed for uuid " << _uuid.toString());
    return DBNodeH();
  }
}

DBNodeH DBNodeH::operator[](const String &key) {
  CoreContext *ctx = CoreContext::instance();
  Database &db = *(ctx->database());
  DatabaseNode *child = db.getChildByName(_uuid, key);
  if (!child) {
    GVT_DEBUG(DBG_ALWAYS, "DBNodeH[] failed to find key \"" << key << "\" for uuid " << _uuid.toString());
    child = &(ctx->createNode(key).getNode());
  }
  return DBNodeH(child->UUID());
}
void DBNodeH::remove() {
  CoreContext *ctx = CoreContext::instance();
  Database &db = *(ctx->database());

  db.removeItem(_uuid);
}

DBNodeH &DBNodeH::operator+=(DBNodeH child) {
  CoreContext *ctx = CoreContext::instance();
  Database &db = *(ctx->database());
  child.setParentUUID(UUID());
  db.addChild(UUID(), &(child.getNode()));
  child.propagateUpdate();
  return *this;
}

DBNodeH &DBNodeH::operator=(Variant val) {
  setValue(val);
  return *this;
}

bool DBNodeH::operator==(const Variant val) { return value() == val; }

DBNodeH::operator bool() const { return !_uuid.isNull(); }

Uuid DBNodeH::UUID() {
  DatabaseNode &n = getNode();
  return n.UUID();
}

String DBNodeH::name() {
  DatabaseNode &n = getNode();
  return n.name();
}

Uuid DBNodeH::parentUUID() {
  DatabaseNode &n = getNode();
  return n.parentUUID();
}

Variant DBNodeH::value() {
  DatabaseNode &n = getNode();
  return n.value();
}

void DBNodeH::setUUID(Uuid uuid) {
  _uuid = uuid;
  DatabaseNode &n = getNode();
  n.setUUID(uuid);
}

void DBNodeH::setName(String name) {
  DatabaseNode &n = getNode();
  n.setName(name);
}

void DBNodeH::setParentUUID(Uuid parentUUID) {
  DatabaseNode &n = getNode();
  n.setParentUUID(parentUUID);
}

void DBNodeH::setValue(Variant value) {
  DatabaseNode &n = getNode();
  n.setValue(value);
}

void DBNodeH::propagateUpdate() {
  DatabaseNode &n = getNode();
  n.propagateUpdate();
}

void DBNodeH::connectValueChanged(const void *receiver, const char *method) {
  GVT_DEBUG(DBG_ALWAYS, "gvt::core::DBNodeH::connectValueChanged not implemented");
}

void DBNodeH::connectChildChanged(const void *receiver, const char *method) {
  GVT_DEBUG(DBG_ALWAYS, "gvt::core::DBNodeH::connectChildChanged not implemented");
}

Vector<DBNodeH> DBNodeH::getChildren() {
  CoreContext *ctx = CoreContext::instance();
  Database &db = *(ctx->database());
  Vector<DatabaseNode *> children = db.getChildren(UUID());
  Vector<DBNodeH> result;
  for (int i = 0; i < children.size(); i++) result.push_back(DBNodeH(children[i]->UUID()));
  return result;
}
