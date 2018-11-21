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
#ifndef GVT_RENDER_DATA_PRIMITIVES_VOLUME_H
#define GVT_RENDER_DATA_PRIMITIVES_VOLUME_H

#include <gvt/render/data/primitives/Data.h>
#include <gvt/render/data/primitives/TransferFunction.h>
#include <gvt/render/data/scene/Light.h>

#include <vector>
namespace gvt {
namespace render {
namespace data {
namespace primitives {


struct griddata {
    int gridid;
    float origin[3];
    float spacing[3];
    int counts[3];
    float * samples;
};

class Volume : public Data {
public:
  Volume();
  Volume(const Volume &);
  ~Volume();
  Box3D boundingBox;
  Box3D *getBoundingBox();
  void SetBoundingBox(glm::vec3 lo, glm::vec3 hi) { boundingBox = Box3D(lo,hi);};
  enum VoxelType // a short list of possible types
  { DOUBLE,
    FLOAT,
    UCHAR,
    SHORT,
    INT };
  float *GetSamples() { return floatsamples; }
  VoxelType GetVoxelType() { return voxtype; }
  void SetVoxelType(VoxelType vtype) { voxtype = vtype; }
  double GetSamplingRate() { return samplingrate; }
  void SetSamplingRate(double rate) { samplingrate = rate; }
  void SetSamples(float *samples) { floatsamples = samples; };
  void SetCounts(int countx, int county, int countz) {
    counts = { countx, county, countz };
    return;
  };
  void GetCounts(glm::vec3 &counts);
  void SetOrigin(float x, float y, float z) { origin = { x, y, z }; };
  void GetDeltas(glm::vec3 &del);
  void SetDeltas(float delx, float dely, float delz);
  // void SetDeltas(glm::vec3 &spaces) {
  // spacing.x = spaces.x;
  // spacing.y=spaces.y;
  // spacing.z=spaces.z;
  // std::cout << " spc " << spacing.x << " " << spacing.y << " " << spacing.z << std::endl;}
  void GetGlobalOrigin(glm::vec3 &origin);
  void SetTransferFunction(TransferFunction *tf) { transfunction = tf; };
  void GetTransferFunction(TransferFunction *tf) { tf = transfunction; };
  TransferFunction *GetTransferFunction() { return transfunction; }
  void SetSlices(int n, glm::vec4 *s);
  void GetSlices(int &n, glm::vec4 *s) {
    n = n_slices;
    s = slices;
  }
  void SetAMRLevels(int n) { numberoflevels = n;};
  void SetAMRGrids(int n) { totalnumberofgrids = n;};
  void AddAMRGrid(int gid, float* origin, float* spacing, int* counts, float* samples);
  void AddAMRGrid(griddata gd);
  void SetIsovalues(int n, float *values);
  void GetIsovalues(int &n, float *values) {
    n = n_isovalues;
    values = isovalues;
  }
  bool is_AMR() { return AMR;};

  virtual std::shared_ptr<Data> getData() { return std::shared_ptr<Data>(this); }

protected:
  glm::vec4 *slices;
  glm::vec3 counts;
  glm::vec3 origin;
  glm::vec3 spacing;
  TransferFunction *transfunction;
  int n_slices;
  float *isovalues;
  int n_isovalues;
  bool AMR;

private:
  VoxelType voxtype;
  double samplingrate;
  glm::vec3 deltas;
  unsigned char *samples;
  float *floatsamples;
  int numberoflevels;
  int totalnumberofgrids;
  std::vector<int> gridsperlevel; 
  std::vector<griddata> gridvector;


};
} // namespace primitives
} // namespace data
} // namespace render
} // namespace gvt
#endif
