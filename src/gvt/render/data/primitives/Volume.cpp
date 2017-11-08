
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
//  shortsamples = other.shortsamples;
//  intsamples = other.intsamples;
//  theOSPVolume = other.theOSPVolume;
//  theOSPData = other.theOSPData;
}
void gvt::render::data::primitives::Volume::SetDeltas(float delx, float dely, float delz) {
  spacing = { delx, dely, delz };
}
// void gvt::render::data::primitives::Volume::SetDeltas(glm::vec3 &del) { spacing = del;}
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
