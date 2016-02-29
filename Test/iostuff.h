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
/* This application tests the OSPRay rendering engine and evaluates some performance
 * */

#include <ply.h>
#include <glob.h>
#include <iostream>
#include <sys/stat.h>
#include <cstdint>
#include <vector>

// file writer right out of ospray example code.
void writePPM(const char *fileName, const int sizeX, const int sizeY, const uint32_t *pixel) {
  FILE *file = fopen(fileName, "wb");
  fprintf(file, "P6\n%i %i\n255\n", sizeX, sizeY);
  unsigned char *out = (unsigned char *)alloca(3 * sizeX);
  for (int y = 0; y < sizeY; y++) {
    const unsigned char *in = (const unsigned char *)&pixel[(sizeY - 1 - y) * sizeX];
    for (int x = 0; x < sizeX; x++) {
      out[3 * x + 0] = in[4 * x + 0];
      out[3 * x + 1] = in[4 * x + 1];
      out[3 * x + 2] = in[4 * x + 2];
    }
    fwrite(out, 3 * sizeX, sizeof(char), file);
  }
  fprintf(file, "\n");
  fclose(file);

  std::string alphaName(fileName);
  alphaName.resize(alphaName.length() - 4); // remove ".ppm"
  alphaName.append("_alpha.pgm");

  file = fopen(alphaName.c_str(), "wb");
  fprintf(file, "P5\n%i %i\n255\n", sizeX, sizeY);
  for (int y = 0; y < sizeY; y++) {
    const unsigned char *in = (const unsigned char *)&pixel[(sizeY - 1 - y) * sizeX];
    for (int x = 0; x < sizeX; x++)
      out[x] = in[4 * x + 3];
    fwrite(out, sizeX, sizeof(char), file);
  }
  fprintf(file, "\n");
  fclose(file);
}
// definitions used in ply file reader
typedef struct Vertex {
  float x, y, z;
  float nx, ny, nz;
  void *other_props;
} Vertex;

typedef struct Face {
  unsigned char nverts;
  int *verts;
  void *other_props;
} Face;

PlyProperty vert_props[] = {
  /* list of property information for a vertex */
  { "x", Float32, Float32, offsetof(Vertex, x), 0, 0, 0, 0 },
  { "y", Float32, Float32, offsetof(Vertex, y), 0, 0, 0, 0 },
  { "z", Float32, Float32, offsetof(Vertex, z), 0, 0, 0, 0 },
  { "nx", Float32, Float32, offsetof(Vertex, nx), 0, 0, 0, 0 },
  { "ny", Float32, Float32, offsetof(Vertex, ny), 0, 0, 0, 0 },
  { "nz", Float32, Float32, offsetof(Vertex, nz), 0, 0, 0, 0 },
};

PlyProperty face_props[] = {
  /* list of property information for a face */
  { "vertex_indices", Int32, Int32, offsetof(Face, verts), 1, Uint8, Uint8, offsetof(Face, nverts) },
};

// determine if file is a directory
bool isdir(const char *path) {
  struct stat buf;
  stat(path, &buf);
  return S_ISDIR(buf.st_mode);
}
// determine if a file exists
bool file_exists(const char *path) {
  struct stat buf;
  return (stat(path, &buf) == 0);
}
/*** search a directory for files named *.ply and return a vector containing the full path to
 * each one.
 **/
std::vector<std::string> findply(const std::string dirname) {
  glob_t result;
  std::string exp = dirname + "/*.ply";
  glob(exp.c_str(), GLOB_TILDE, NULL, &result);
  std::vector<std::string> ret;
  for (int i = 0; i < result.gl_pathc; i++) {
    ret.push_back(std::string(result.gl_pathv[i]));
  }
  globfree(&result);
  return ret;
}
/*** read a ply file and stuff the data into an ospgeometry object. Expects a triangle
 * mesh. If the ply file contains non triangular faces then bad things will probably happen.
 * */
void ReadPlyData(std::string filename, float *&vertexarray, float *&colorarray, int32_t *&indexarray, int &nverts,
                 int &nfaces) {
  FILE *InputFile;
  PlyFile *in_ply;
  std::string elem_name;
  int elem_count, i, j;
  int32_t *index;
  Vertex *vert;
  Face *face;
  // default color of vertex
  float color[] = { 0.5f, 0.5f, 1.0f, 1.0f };
  InputFile = fopen(filename.c_str(), "r");
  in_ply = read_ply(InputFile);
  for (i = 0; i < in_ply->num_elem_types; i++) {
    elem_name = std::string(setup_element_read_ply(in_ply, i, &elem_count));
    if (elem_name == "vertex") {
      nverts = elem_count;
      vertexarray = (float *)malloc(3 * nverts * sizeof(float)); // allocate vertex array
      colorarray = (float *)malloc(4 * nverts * sizeof(float));  // allocate color array
      setup_property_ply(in_ply, &vert_props[0]);
      setup_property_ply(in_ply, &vert_props[1]);
      setup_property_ply(in_ply, &vert_props[2]);
      vert = (Vertex *)malloc(sizeof(Vertex));
      for (j = 0; j < elem_count; j++) {
        get_element_ply(in_ply, (void *)vert);
        vertexarray[3 * j] = vert->x;
        vertexarray[3 * j + 1] = vert->y;
        vertexarray[3 * j + 2] = vert->z;
        colorarray[4 * j] = color[0];
        colorarray[4 * j + 1] = color[1];
        colorarray[4 * j + 2] = color[2];
        colorarray[4 * j + 3] = color[3];
      }
    } else if (elem_name == "face") {
      nfaces = elem_count;
      indexarray = (int32_t *)malloc(3 * nfaces * sizeof(int32_t));
      setup_property_ply(in_ply, &face_props[0]);
      face = (Face *)malloc(sizeof(Face));
      for (j = 0; j < elem_count; j++) {
        get_element_ply(in_ply, (void *)face);
        indexarray[3 * j] = face->verts[0];
        indexarray[3 * j + 1] = face->verts[1];
        indexarray[3 * j + 2] = face->verts[2];
      }
    }
  }
  close_ply(in_ply);
}
