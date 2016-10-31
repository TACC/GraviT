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


#ifndef GVT_RENDER_DATA_DOMAIN_READER_PLY_READER_H
#define GVT_RENDER_DATA_DOMAIN_READER_PLY_READER_H

#include <gvt/render/data/Primitives.h>

#include <ply.h>
#include <glob.h>
#include <sys/stat.h>

 namespace gvt {
 namespace render {
 namespace data {
 namespace domain {
 namespace reader {
 /// read ply formatted geometry data
 /** read ply format files and return a Mesh object
 */
 class PlyReader {
 public:
   PlyReader(const std::string filename);
   virtual ~PlyReader();

   typedef struct Vertex {
     float x, y, z;
     float nx, ny, nz;
     void *other_props; /* other properties */
   } Vertex;

   typedef struct Face {
     unsigned char nverts; /* number of vertex indices in list */
     int *verts;           /* vertex index list */
     void *other_props;    /* other properties */
   } Face;

   bool file_exists(const char *path) {
     struct stat buf;
     return (stat(path, &buf) == 0);
   }

   bool isdir(const char *path) {
     struct stat buf;
     stat(path, &buf);
     return S_ISDIR(buf.st_mode);
   }

   gvt::core::Vector<std::string> findply(const std::string dirname) {
     glob_t result;
     std::string exp = dirname + "/*.ply";
     glob(exp.c_str(), GLOB_TILDE, NULL, &result);
     gvt::core::Vector<std::string> ret;
     for (int i = 0; i < result.gl_pathc; i++) {
       ret.push_back(std::string(result.gl_pathv[i]));
     }
     globfree(&result);
     return ret;
   }

   gvt::core::Vector<gvt::render::data::primitives::Mesh*>& getMeshes(){
     return meshes;
   }

 private:
    Vertex **vlist;
    Face **flist;

   gvt::core::Vector<gvt::render::data::primitives::Mesh*> meshes;


 };
 }
 }
 }
 }
 }
#endif /* GVT_RENDER_DATA_DOMAIN_READER_PLY_READER_H */
