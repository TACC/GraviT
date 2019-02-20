
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
#include <gvt/render/data/primitives/Volume.h>

gvt::render::data::primitives::Volume::Volume() {
  n_slices = 0;
  counts = { 0, 0, 0 };
  n_isovalues = 0;
  slices = NULL;
  isovalues = NULL;
  counts = { 1, 1, 1 };
  origin = { 0.0, 0.0, 0.0 };
  spacing = { 1.0, 1.0, 1.0 };
  AMR = false;
}

gvt::render::data::primitives::Volume::Volume(const Volume& other) {
  n_slices = other.n_slices;
  n_isovalues = other.n_isovalues;
  slices = other.slices;
  isovalues = other.isovalues;
  counts = other.counts;
  origin = other.origin;
  spacing = other.spacing;

  boundingBox = other.boundingBox;
  transfunction = other.transfunction;
  voxtype = other.voxtype;
  samplingrate = other.samplingrate;
  deltas = other.deltas;
  samples = other.samples;
  floatsamples = other.floatsamples;
  AMR = other.AMR;
}
void gvt::render::data::primitives::Volume::SetDeltas(float delx, float dely, float delz) {
  spacing = { delx, dely, delz };
}
void gvt::render::data::primitives::Volume::AddAMRGrid(gvt::render::data::primitives::griddata gd) { 
    gvt::render::data::primitives::griddata data;
    data.gridid = gd.gridid;
    data.origin[0] = gd.origin[0];
    data.origin[1] = gd.origin[1];
    data.origin[2] = gd.origin[2];
    data.spacing[0] = gd.spacing[0];
    data.spacing[1] = gd.spacing[1];
    data.spacing[2] = gd.spacing[2];
    data.counts[0] = gd.counts[0];
    data.counts[1] = gd.counts[1];
    data.counts[2] = gd.counts[2];
    data.samples = gd.samples;
    numberofgridsinvolume +=1;
    gridvector.push_back(data);
}

void gvt::render::data::primitives::Volume::AddAMRGrid(int gridid, int level, float *orig, float *spac, int *counts, float *samp) {
    gvt::render::data::primitives::griddata data;
    std::cerr << "gvt:Volume:AddAMRGrid: gid " << gridid << " level " << level << std::endl;
    data.gridid = gridid;
    data.level = level;
    data.origin[0] = orig[0];
    data.origin[1] = orig[1];
    data.origin[2] = orig[2];
    data.spacing[0] = spac[0];
    data.spacing[1] = spac[1];
    data.spacing[2] = spac[2];
    std::cerr << "gvt:volume:AddAMRGrid: counts " << counts[0] << " " << counts[1] << " " << counts[2] << std::endl;
    int numsamples = counts[0]*counts[1]*counts[2];
    std::cerr << "gvt:volume:AddAMRGrid: numsamples " << numsamples << std::endl;
    data.samples = new float[numsamples];
    for(int i = 0;i<numsamples;i++) 
        data.samples[i] = samp[i];
    data.counts[0] = counts[0]-1;
    data.counts[1] = counts[1]-1;
    data.counts[2] = counts[2]-1;
    numberofgridsinvolume +=1;
    gridvector.push_back(data);
    std::map<int,int>::iterator lngit = lng.find(level);
    if(lngit == lng.end()) {
        lng.insert(std::make_pair(level,1));
    } else {
        lngit->second = lngit->second + 1;
    }
}
gvt::render::data::primitives::griddata gvt::render::data::primitives::Volume::GetAMRGrid(int gid)
{
    //std::cerr << " gid " << gid << " gridvector size " << gridvector.size() << std::endl;
    //std::cerr << gridvector[gid].origin[0] << " " << gridvector[gid].origin[1] << " " << gridvector[gid].origin[2] << std::endl;
       //return gridvector[gid];
       return gridvector.at(gid);
}

void gvt::render::data::primitives::Volume::GetDeltas(glm::vec3 &del) { del = spacing; }
void gvt::render::data::primitives::Volume::GetCounts(glm::vec3 &cnts) { cnts = counts; }
void gvt::render::data::primitives::Volume::GetGlobalOrigin(glm::vec3 &orig) { orig = origin; }
gvt::render::data::primitives::Volume::~Volume() {}
gvt::render::data::primitives::Box3D *gvt::render::data::primitives::Volume::getBoundingBox() { return &boundingBox; }
void gvt::render::data::primitives::Volume::SetIsovalues(int n, float *vals) {
  n_isovalues = n;
  isovalues = new float[n_isovalues];
  for (int i = 0; i < n_isovalues; i++) {
    isovalues[i] = vals[i];
  }
}
