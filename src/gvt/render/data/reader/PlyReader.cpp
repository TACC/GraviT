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

#include <gvt/render/data/reader/PlyReader.h>
#include <gvt/render/RenderContext.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

using namespace gvt::render::data::domain::reader;
using namespace gvt::render::data::primitives;
using namespace std;

PlyProperty vert_props[] = {
  /* list of property information for a vertex */
  { "x", Float32, Float32, offsetof(PlyReader::Vertex, x), 0, 0, 0, 0 },
  { "y", Float32, Float32, offsetof(PlyReader::Vertex, y), 0, 0, 0, 0 },
  { "z", Float32, Float32, offsetof(PlyReader::Vertex, z), 0, 0, 0, 0 },
  { "nx", Float32, Float32, offsetof(PlyReader::Vertex, nx), 0, 0, 0, 0 },
  { "ny", Float32, Float32, offsetof(PlyReader::Vertex, ny), 0, 0, 0, 0 },
  { "nz", Float32, Float32, offsetof(PlyReader::Vertex, nz), 0, 0, 0, 0 },
};

PlyProperty face_props[] = {
  /* list of property information for a face */
  { "vertex_indices", Int32, Int32, offsetof(PlyReader::Face, verts), 1, Uint8, Uint8,
    offsetof(PlyReader::Face, nverts) },
};

PlyReader::PlyReader(std::string rootdir) {

  // mess I use to open and read the ply file with the c utils I found.
  PlyFile *in_ply;
  Vertex *vert;
  Face *face;
  int elem_count, nfaces, nverts;
  int i, j, k;
  float xmin, ymin, zmin, xmax, ymax, zmax;
  char *elem_name;
  ;
  FILE *myfile;
  char txt[16];
  std::string temp;
  std::string filename, filepath;

  gvt::render::RenderContext *cntxt = gvt::render::RenderContext::instance();
  if (cntxt == NULL) {
    std::cout << "Something went wrong initializing the context" << std::endl;
    exit(0);
  }
  gvt::core::DBNodeH root = cntxt->getRootNode();
  gvt::core::DBNodeH dataNodes = root["Data"];
  gvt::core::DBNodeH instNodes = root["Instances"];

  // Enzo isosurface...
  if (!file_exists(rootdir.c_str())) {
    cout << "File \"" << rootdir << "\" does not exist. Exiting." << endl;
    exit(0);
  }

  if (!isdir(rootdir.c_str())) {
    cout << "File \"" << rootdir << "\" is not a directory. Exiting." << endl;
    exit(0);
  }
  vector<string> files = findply(rootdir);
  if (files.empty()) {
    cout << "Directory \"" << rootdir << "\" contains no .ply files. Exiting." << endl;
    exit(0);
  }
  // read 'em
  vector<string>::const_iterator file;
  // for (k = 0; k < 8; k++) {
  for (file = files.begin(), k = 0; file != files.end(); file++, k++) {
// if defined, ply blocks load are divided across available mpi ranks
// Each block will be loaded by a single mpi rank and a mpi rank can read multiple blocks
#ifdef DOMAIN_PER_NODE
    if (!((k >= MPI::COMM_WORLD.Get_rank() * DOMAIN_PER_NODE) &&
          (k < MPI::COMM_WORLD.Get_rank() * DOMAIN_PER_NODE + DOMAIN_PER_NODE)))
      continue;
#endif

// if all ranks read all ply blocks, one has to create the db node which is then broadcasted.
// if not, since each block will be loaded by only one mpi, this mpi rank will create the db node
#ifndef DOMAIN_PER_NODE
    if (MPI::COMM_WORLD.Get_rank() == 0)
#endif
      gvt::core::DBNodeH PlyMeshNode = cntxt->addToSync(cntxt->createNodeFromType("Mesh", *file, dataNodes.UUID()));

#ifndef DOMAIN_PER_NODE
    cntxt->syncContext();
    gvt::core::DBNodeH PlyMeshNode = dataNodes.getChildren()[k];
#endif
    filepath = *file;
    myfile = fopen(filepath.c_str(), "r");
    in_ply = read_ply(myfile);
    for (i = 0; i < in_ply->num_elem_types; i++) {
      elem_name = setup_element_read_ply(in_ply, i, &elem_count);
      temp = elem_name;
      if (temp == "vertex") {
        vlist = (Vertex **)malloc(sizeof(Vertex *) * elem_count);
        nverts = elem_count;
        setup_property_ply(in_ply, &vert_props[0]);
        setup_property_ply(in_ply, &vert_props[1]);
        setup_property_ply(in_ply, &vert_props[2]);
        for (j = 0; j < elem_count; j++) {
          vlist[j] = (Vertex *)malloc(sizeof(Vertex));
          get_element_ply(in_ply, (void *)vlist[j]);
        }
      } else if (temp == "face") {
        flist = (Face **)malloc(sizeof(Face *) * elem_count);
        nfaces = elem_count;
        setup_property_ply(in_ply, &face_props[0]);
        for (j = 0; j < elem_count; j++) {
          flist[j] = (Face *)malloc(sizeof(Face));
          get_element_ply(in_ply, (void *)flist[j]);
        }
      }
    }
    close_ply(in_ply);
    // smoosh data into the mesh object

      Material *m = new Material();
      Mesh *mesh = new Mesh(m);
      vert = vlist[0];
      xmin = vert->x;
      ymin = vert->y;
      zmin = vert->z;
      xmax = vert->x;
      ymax = vert->y;
      zmax = vert->z;

      for (i = 0; i < nverts; i++) {
        vert = vlist[i];
        xmin = MIN(vert->x, xmin);
        ymin = MIN(vert->y, ymin);
        zmin = MIN(vert->z, zmin);
        xmax = MAX(vert->x, xmax);
        ymax = MAX(vert->y, ymax);
        zmax = MAX(vert->z, zmax);
        mesh->addVertex(glm::vec3(vert->x, vert->y, vert->z));
      }
      glm::vec3 lower(xmin, ymin, zmin);
      glm::vec3 upper(xmax, ymax, zmax);
      Box3D *meshbbox = new gvt::render::data::primitives::Box3D(lower, upper);
      // add faces to mesh
      for (i = 0; i < nfaces; i++) {
        face = flist[i];
        mesh->addFace(face->verts[0] + 1, face->verts[1] + 1, face->verts[2] + 1);
      }
      mesh->generateNormals();

      PlyMeshNode["file"] = string(filepath);
      PlyMeshNode["bbox"] = (unsigned long long)meshbbox;
      PlyMeshNode["ptr"] = (unsigned long long)mesh;

      gvt::core::DBNodeH loc = cntxt->createNode("rank", MPI::COMM_WORLD.Get_rank());
      PlyMeshNode["Locations"] += loc;

      cntxt->addToSync(PlyMeshNode);

      meshes.push_back(mesh);

  }

  cntxt->syncContext();


}

PlyReader::~PlyReader() {}
