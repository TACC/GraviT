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
#include <map>
#include <iterator>
namespace gvt {
namespace render {
namespace data {
namespace primitives {


struct griddata {
    int gridid;      // global grid index. Index of this grid in total number of grids.
    int level;       // the level of refinement of this grid. 
    float origin[3]; // origin of this grid in world coords
    float spacing[3];// cell spacing of this grid
    int counts[3];   // dimensions of this grid
    float bounds[6]; // bounds of this  grid.
    float * samples; // pointer to scalar at grid points
    std::vector<int> subgrids; // vector of subgrid indices of this grid 
};

/**
 * A Volume consists of a collection of nested grids. There is one level 0 grid and
 * any number of subgrids. The level 0 grid contains the global origin, counts, and 
 * spacing. It maintains a level 0 grid structure which in turn contains a vector of
 * subgrids. The whole collection forms a tree with grids on each level of the tree.
 * All necessary information about a grid including the sample data is contained in
 * the griddata structure. 
 */
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
  void SetAMRNumberOfGridsInVolume(int n) {numberofgridsinvolume = n;};
  int GetAMRNumberOfGridsInVolume() { return numberofgridsinvolume;};
  int GetAMRNumberOfLevels() { return numberoflevels;};
  void SetAMRGrids(int n) { totalnumberofgrids = n;};
  void SetAMRlng(int level,int count) {lng.insert(std::make_pair(level,count));};
  std::map<int,int> GetAmrlng() { return lng;};
  int GetAMRTotalNumberOfGrids() { return totalnumberofgrids;};
  int GetAMRNumberOfGridsInLevel(int level) {return gridsperlevel[level];};
  void AddAMRGrid(int gid, int level, float* origin, float* spacing, int* counts, float* samples);
  griddata GetAMRGrid(int gid); 
  void AddAMRGrid(griddata gd);
  void SetAMRBounds(double b[]) { bounds[0] = b[0];
                                  bounds[1] = b[1];
                                  bounds[2] = b[2];
                                  bounds[3] = b[3];
                                  bounds[4] = b[4];
                                  bounds[5] = b[5]; };
  float * GetAMRBounds() { float *b = new float[6]; 
                            b[0] = bounds[0]; 
                            b[1] = bounds[1]; 
                            b[2] = bounds[2]; 
                            b[3] = bounds[3]; 
                            b[4] = bounds[4]; 
                            b[5] = bounds[5]; 
                            return b;
  };
  void SetIsovalues(int n, float *values);
  void GetIsovalues(int &n, float *values) {
    n = n_isovalues;
    values = isovalues;
  }
  bool is_AMR() { return AMR;};
  void SetAMRTrue() { AMR = true;};
  void SetAMRFalse() {AMR = false;};

  virtual std::shared_ptr<Data> getData() { return std::shared_ptr<Data>(this); }

protected:
  glm::vec4 *slices;
  glm::vec3 counts; // if AMR this contains dims of level0 grid
  glm::vec3 origin; // if AMR this contains global origin.
  glm::vec3 spacing;// if AMR this contains spacing of level0 grid
  TransferFunction *transfunction;
  int n_slices;
  float *isovalues;
  int n_isovalues;
  bool AMR;

private:
  VoxelType voxtype;
  double samplingrate;
  double bounds[6];
  glm::vec3 deltas;
  unsigned char *samples;
  float *floatsamples;
  int numberoflevels; // for amr number of levels in this volume
  int totalnumberofgrids; // for amr total number of grids in this volume
  int numberofgridsinvolume; // number of grids in this volume
  std::vector<int> gridsperlevel; // for amr number of grids per level
  std::map<int,int> lng; // map of levels nuber of grids
  std::vector<griddata> gridvector; // vector of grids in this volume


};
} // namespace primitives
} // namespace data
} // namespace render
} // namespace gvt
#endif
