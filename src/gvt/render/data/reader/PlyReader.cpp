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

#include <gvt/render/api/api.h>
#include <gvt/render/cntx/rcontext.h>
#include <gvt/render/data/reader/PlyReader.h>

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
  { "red", Uint8, Uint8, offsetof(PlyReader::Vertex, cx), 0, 0, 0, 0 },
  { "green", Uint8, Uint8, offsetof(PlyReader::Vertex, cy), 0, 0, 0, 0 },
  { "blue", Uint8, Uint8, offsetof(PlyReader::Vertex, cz), 0, 0, 0, 0 }
};

PlyProperty face_props[] = {
  /* list of property information for a face */
  { "vertex_indices", Int32, Int32, offsetof(PlyReader::Face, verts), 1, Uint8, Uint8,
    offsetof(PlyReader::Face, nverts) },
};

PlyReader::PlyReader(std::string rootdir, bool dist) {
  // gvt::comm::communicator &comm = gvt::comm::communicator::instance();
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

  auto &db = cntx::rcontext::instance();

  //  gvt::core::DBNodeH root = cntxt->getRootNode();
  //  gvt::core::DBNodeH dataNodes = root["Data"];
  //  gvt::core::DBNodeH instNodes = root["Instances"];

  auto &dataNodes = db.getUnique("Data");
  auto &instNodes = db.getUnique("Instance");

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

    std::string meshname = "Mesh" + std::to_string(k);

    api::createMesh(meshname);

    filepath = *file;
    myfile = fopen(filepath.c_str(), "r");
    in_ply = read_ply(myfile);
    bool has_color = false;
    for (i = 0; i < in_ply->num_elem_types; i++) {
      elem_name = setup_element_read_ply(in_ply, i, &elem_count);
      temp = elem_name;
      if (temp == "vertex") {
        vlist = (Vertex **)malloc(sizeof(Vertex *) * elem_count);
        nverts = elem_count;
        setup_property_ply(in_ply, &vert_props[0]);
        setup_property_ply(in_ply, &vert_props[1]);
        setup_property_ply(in_ply, &vert_props[2]);
        if (in_ply->elems[i]->nprops > 5) {
          has_color = true;
          setup_property_ply(in_ply, &vert_props[3]);
          setup_property_ply(in_ply, &vert_props[4]);
          setup_property_ply(in_ply, &vert_props[5]);
        }
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

    Material *m = has_color ? nullptr : new Material();
    std::shared_ptr<Mesh> mesh = std::make_shared<Mesh>(m);

    std::shared_ptr<float> vtx(new float[nverts * 3], std::default_delete<float[]>());
    std::shared_ptr<float> color(new float[nverts * 3], std::default_delete<float[]>());
    std::shared_ptr<unsigned> fc(new unsigned[nverts * 3], std::default_delete<unsigned[]>());

    float *avtx = vtx.get();
    float *acolor = color.get();
    unsigned *afc = fc.get();

    for (int i = 0; i < nverts; i++) {
      avtx[i * 3 + 0] = vlist[i]->x;
      avtx[i * 3 + 1] = vlist[i]->y;
      avtx[i * 3 + 2] = vlist[i]->z;

      if (has_color) {
        acolor[i * 3 + 0] = float(vlist[i]->cx) / 255.f;
        acolor[i * 3 + 0] = float(vlist[i]->cy) / 255.f;
        acolor[i * 3 + 0] = float(vlist[i]->cz) / 255.f;
      }
    }

    for (int f = 0; f < nverts; f++) {
      afc[i * 3 + 0] = flist[i]->verts[0];
      afc[i * 3 + 1] = flist[i]->verts[1];
      afc[i * 3 + 2] = flist[i]->verts[2];
    }

    api::addMeshVertices(meshname, nverts, avtx);
    api::addMeshTriangles(meshname, nfaces, afc);
    if (has_color) api::addMeshVertexColor(meshname, nverts, acolor);
    api::finishMesh(meshname);
  }
}

PlyReader::~PlyReader() {}
