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
#include "gvt/core/Context.h"
#include "gvt/core/Database.h"
#include "gvt/core/DatabaseNode.h"
#include "gvt/core/Debug.h"

using namespace gvt::core;

int main(int argc, char **argv) {
  Context &ctx = *Context::singleton();
  Database &db = *ctx.database();

  GVT_DEBUG(DBG_ALWAYS, "Created database with root node uuid " << ctx.getRootNode().UUID().toString());

  DBNodeH session = ctx.createNodeFromType(String("Session"), ctx.getRootNode().UUID());
  DBNodeH renderable = ctx.createNodeFromType(String("Renderable3DGrid"), ctx.getRootNode().UUID());

  GVT_DEBUG(DBG_ALWAYS, "Created session with node uuid " << session.UUID().toString());
  GVT_DEBUG(DBG_ALWAYS, "Created renderable with node uuid " << renderable.UUID().toString());

  session["foo"] = String("foo");
  renderable["bar"] = 5;

  db.printTree(ctx.getRootNode().UUID());

  return 0;
}