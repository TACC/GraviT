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
 * aspects of the library. The results can be compared to similar tests of GraviT to
 * determine the impact of the GraviT overhead.
 *
 * command line args:
 *
 * -i <file> input file name. Should be a .ply file. If a directory the directory is
 *           searched for .ply files and each is added as a triangle mesh.
 * -o <file> causes a .ppm file to be written with name file.
 * -bench <warmxbench> number of warmup and bench frames respectively (-bench 10x100)
 * -geom <widthxheight> width and height of image. (-geom 1920x1080)
 * -cp <x,y,z> list of coordinates for camera position in world coords.
 * -cd <x,y,z> direction vector where camera is looking (not focal point)
 * -cu <x,y,z> camera up vector.
 * -fov <angle> vertical field of view angle of the camera.
 * -renderer <ren> name of renerer, (obj, scivis, raytracer, ao, ... )
 * -ld <x,y,z> direction light goes. direction vector similar to camera direction.
 *
 *   for example
 *    bin/osptest -i $WORK/DAVEDATA/EnzoPlyData -o spoot.ppm -cp 512,512,4096 -cd 0,0,-1 -fov 25.0 -ld 0,0,-1 -renderer obj -geom 1920x1080
 *
 * Tests Performed:
 * 	1) rendering blank screen with no geometry.
 * 	2) rendering large geometry data (Enzo isosurface)
 * */

#include "../iostuff.h"
#include "ospray/ospray.h"


#include <sstream>
#include <time.h>
#include "../timer.h"

int main(int argc, const char **argv) {
  // default values
  int width = 1920;
  int height = 1080;
  int warmupframes = 1;
  int benchframes = 10;
  // timer stuff
  my_timer_t startTime, endTime;
  double rendertime = 0.0;
  double warmupframetime = 0.0;
  double iotime = 0.0;
  double modeltime = 0.0;
  // empty vertex list
  float *vertexarray;
  float *colorarray;
  int32_t *indexarray;
  int nverts, nfaces;
  int numtriangles = 0;
  float vertex[1];
  float color[1];
  int32_t index[1];
  // file related things
  std::string filepath("");
  std::string outputfile("");
  std::string renderertype("obj");
  // initialize ospray
  ospInit(&argc, argv);
  OSPGeometry mesh;
  OSPModel world = ospNewModel();
  OSPCamera camera = ospNewCamera("perspective");
  // default camera and light settings
  osp::vec3f cam_pos = { -1000.f, 0.f, -1000.f };
  osp::vec3f cam_up = { 0.f, 1.f, 0.f };
  osp::vec3f cam_view = { 0.1f, 0.0f, 0.1f };
  osp::vec3f light_dir = { 0., 0., 1.0 };
  float cam_fovy = 50.0;
  float lightdirection[] = { 0., 0., 1.0 };
  // parse the command line
  if ((argc < 2)) {
    // no input so render default empty image.
  } else {
    // parse the input
    for (int i = 1; i < argc; i++) {
      const std::string arg = argv[i];
      if (arg == "-i") { // set the path to the input file
        filepath = argv[++i];
        if (!file_exists(filepath.c_str())) {
          std::cout << "File \"" << filepath << "\" does not exist. Exiting." << std::endl;
          return 0;
          // test to see if the file is a directory
        } else if (isdir(filepath.c_str())) { // read all .ply files in a directory
          std::vector<std::string> files = findply(filepath);
          if (!files.empty()) { // parse the files and add the meshes.
            std::vector<std::string>::const_iterator file;
            for (file = files.begin(); file != files.end(); file++) {
              timeCurrent(&startTime);
              ReadPlyData(*file, vertexarray, colorarray, indexarray, nverts, nfaces);
              timeCurrent(&endTime);
              iotime += timeDifferenceMS(&startTime, &endTime);
              std::cout << " file " << timeDifferenceMS(&startTime, &endTime)<< std::endl;
              timeCurrent(&startTime);
              mesh = ospNewGeometry("triangles");
              OSPData data = ospNewData(nverts, OSP_FLOAT3, vertexarray);
              ospCommit(data);
              ospSetData(mesh, "vertex", data);

              data = ospNewData(nverts, OSP_FLOAT4, colorarray);
              ospCommit(data);
              ospSetData(mesh, "vertex.color", data);

              data = ospNewData(nfaces, OSP_INT3, indexarray);
              ospCommit(data);
              ospSetData(mesh, "index", data);

              ospCommit(mesh);
              ospAddGeometry(world, mesh);
              timeCurrent(&endTime);
              modeltime += timeDifferenceMS(&startTime, &endTime);
              numtriangles += nfaces;
            }
          } else {
            filepath = "";
          }
        } else { // read a single file into a mesh.
          timeCurrent(&startTime);
          ReadPlyData(filepath, vertexarray, colorarray, indexarray, nverts, nfaces);
          timeCurrent(&endTime);
          iotime += timeDifferenceMS(&startTime, &endTime);
          timeCurrent(&startTime);
          mesh = ospNewGeometry("triangles");
          OSPData data = ospNewData(nverts, OSP_FLOAT3, vertexarray);
          ospCommit(data);
          ospSetData(mesh, "vertex", data);

          data = ospNewData(nverts, OSP_FLOAT4, colorarray);
          ospCommit(data);
          ospSetData(mesh, "vertex.color", data);

          data = ospNewData(nfaces, OSP_INT3, indexarray);
          ospCommit(data);
          ospSetData(mesh, "index", data);

          ospCommit(mesh);
          ospAddGeometry(world, mesh);
          timeCurrent(&endTime);
          modeltime += timeDifferenceMS(&startTime, &endTime);
          numtriangles += nfaces;
        }
      } else if (arg == "-bench") { // taken from ospray example
        if (++i < argc) {
          std::string arg2(argv[i]);
          size_t pos = arg2.find("x");
          if (pos != std::string::npos) {
            arg2.replace(pos, 1, " ");
            std::stringstream ss(arg2);
            ss >> warmupframes >> benchframes;
          }
        }
      } else if (arg == "-geom") {
        if (++i < argc) {
          std::string arg2(argv[i]);
          size_t pos = arg2.find("x");
          if (pos != std::string::npos) {
            arg2.replace(pos, 1, " ");
            std::stringstream ss(arg2);
            ss >> width >> height;
          }
        }
      } else if (arg == "-o") {
        outputfile = argv[++i];
      } else if (arg == "-cp") { // set camera position
        if (++i < argc) {
          std::string arg2(argv[i]);
          size_t pos = arg2.find(",");
          if (pos != std::string::npos) {
            arg2.replace(pos, 1, " ");
          }
          pos = arg2.find(",");
          if (pos != std::string::npos) {
            arg2.replace(pos, 1, " ");
          }
          float camx, camy, camz;
          std::stringstream ss(arg2);
          ss >> camx >> camy >> camz;
          cam_pos = { camx, camy, camz };
        }
      } else if (arg == "-cd") { // set camera direction
        if (++i < argc) {
          std::string arg2(argv[i]);
          size_t pos = arg2.find(",");
          if (pos != std::string::npos) {
            arg2.replace(pos, 1, " ");
          }
          pos = arg2.find(",");
          if (pos != std::string::npos) {
            arg2.replace(pos, 1, " ");
          }
          float cdx, cdy, cdz;
          std::stringstream ss(arg2);
          ss >> cdx >> cdy >> cdz;
          cam_view = { cdx, cdy, cdz };
        }
      } else if (arg == "-cu") { // set camera up direction
        if (++i < argc) {
          std::string arg2(argv[i]);
          size_t pos = arg2.find(",");
          if (pos != std::string::npos) {
            arg2.replace(pos, 1, " ");
          }
          pos = arg2.find(",");
          if (pos != std::string::npos) {
            arg2.replace(pos, 1, " ");
          }
          float cux, cuy, cuz;
          std::stringstream ss(arg2);
          ss >> cux >> cuy >> cuz;
          cam_up = { cux, cuy, cuz };
        }
      } else if (arg == "-ld") { // set light direction
        if (++i < argc) {
          std::string arg2(argv[i]);
          size_t pos = arg2.find(",");
          if (pos != std::string::npos) {
            arg2.replace(pos, 1, " ");
          }
          pos = arg2.find(",");
          if (pos != std::string::npos) {
            arg2.replace(pos, 1, " ");
          }
          float ldx, ldy, ldz;
          std::stringstream ss(arg2);
          ss >> ldx >> ldy >> ldz;
          light_dir = { ldx, ldy, ldz };
        }
      } else if (arg == "-fov") { // grab the field of view
        cam_fovy = atof(argv[++i]);
      } else if (arg == "-renderer") {
        renderertype = argv[++i];
      }
    }
  }
  //
  // Create empty data set (dont know if this is necessary or not to do empty
  // screen test) if there is no filename given
  //
  timeCurrent(&startTime);
  if (filepath.empty()) {
    // empty filepath render blank screen 
    mesh = ospNewGeometry("triangles");
    OSPData data = ospNewData(0, OSP_FLOAT3A, vertex);
    ospCommit(data);
    ospSetData(mesh, "vertex", data);

    data = ospNewData(0, OSP_FLOAT4, color);
    ospCommit(data);
    ospSetData(mesh, "vertex.color", data);

    data = ospNewData(0, OSP_INT3, index);
    ospCommit(data);
    ospSetData(mesh, "index", data);

    ospCommit(mesh);
    ospAddGeometry(world, mesh);
  }
  ospSetVec3f(camera, "pos", cam_pos);
  ospSetf(camera, "aspect", width / (float)height);
  ospSetf(camera, "fovy", cam_fovy);
  ospSetVec3f(camera, "dir", cam_view);
  ospSetVec3f(camera, "up", cam_up);
  ospCommit(camera);

  ospCommit(world);
  // framebuffer and renderer
  OSPRenderer renderer = ospNewRenderer(renderertype.c_str());
  ospSet3f(renderer,"bgColor",0.,0.,0.);
  ospSetObject(renderer, "model", world);
  ospSetObject(renderer, "camera", camera);
  ospCommit(renderer);
  // Light
  OSPLight somelight = ospNewLight(renderer, "DirectionalLight");
  ospSetString(somelight,"name","sun");
  ospSet3f(somelight, "color", 1, 1, 1);
  ospSetVec3f(somelight, "direction", light_dir);
  //ospSet1f(somelight,"intensity",250.f);
  ospCommit(somelight);
  OSPData lightArray = ospNewData(1, OSP_OBJECT, &somelight,0);
  // point light does not work for some reason
  //OSPLight ptlight = ospNewLight(renderer,"PointLight");
  //ospSet3f(ptlight,"color",1,1,1);
  //ospSetVec3f(ptlight,"position",cam_pos);
  //ospSet1f(ptlight,"intensity",250.f);
  //ospSet1f(ptlight,"radius",4.f);
  //ospCommit(ptlight);
  //OSPData lightArray = ospNewData(1, OSP_OBJECT, &ptlight,0);
  ospSetData(renderer, "lights", lightArray);
  ospCommit(renderer);
  osp::vec2i framebufferdimensions = { width, height };
  OSPFrameBuffer framebuffer = ospNewFrameBuffer(framebufferdimensions, OSP_FB_RGBA8, OSP_FB_COLOR | OSP_FB_ACCUM);
  ospFrameBufferClear(framebuffer, OSP_FB_COLOR | OSP_FB_ACCUM);
  timeCurrent(&endTime);
  modeltime += timeDifferenceMS(&startTime, &endTime);

  // warmup
  timeCurrent(&startTime);
  for (int frame = 0; frame < warmupframes; frame++)
    ospRenderFrame(framebuffer, renderer, OSP_FB_COLOR | OSP_FB_ACCUM);
  timeCurrent(&endTime);
  warmupframetime = timeDifferenceMS(&startTime, &endTime);
  // benchmark
  timeCurrent(&startTime);
  for (int frame = 0; frame < benchframes; frame++)
    ospRenderFrame(framebuffer, renderer, OSP_FB_COLOR | OSP_FB_ACCUM);
  timeCurrent(&endTime);
  //
  rendertime = timeDifferenceMS(&startTime, &endTime);
  float millionsoftriangles = numtriangles / 1000000.;
  float millisecondsperframe = rendertime / benchframes;
  float framespersecond = (1000 * benchframes) / rendertime;
  // dump out csv of values
  std::cout << renderertype << "," << width << "," << height << "," << warmupframes << "," << benchframes << ","
            << iotime;
  std::cout << "," << modeltime << "," << warmupframetime << "," << millisecondsperframe << "," << framespersecond << std::endl;
  // std::cout << millionsoftriangles << " million triangles" << std::endl;
  // std::cout << "iotime (ms) " << iotime << " modeltime (ms) " << modeltime << std::endl;
  // std::cout << millisecondsperframe << " (ms)/frame " << framespersecond << " fps " << std::endl;
  if (!outputfile.empty()) {
    const uint32_t *fb = (uint32_t *)ospMapFrameBuffer(framebuffer, OSP_FB_COLOR);
    writePPM(outputfile.c_str(), width, height, fb);
  }
  return 0;
}
