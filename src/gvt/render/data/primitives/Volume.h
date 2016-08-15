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

#include <gvt/render/data/primitives/BBox.h>
#include <gvt/render/data/primitives/TransferFunction.h>
#include <gvt/render/data/scene/Light.h>

#include <vector>
namespace gvt {
namespace render {
namespace data {
namespace primitives {

class Volume {
public:
  Volume();
  ~Volume();
  Box3D boundingBox;
  Box3D *getBoundingBox();
  enum DataType
  {
    FLOAT, UCHAR
  };
  //unsigned char *get_samples() ;
  short *GetSamples() {return shortsamples;};
  void SetSamples(short * samples) {shortsamples = samples;return;};
  void SetCounts(int countx, int county, int countz) {counts = {countx,county,countz};return;};
  glm::vec3* GetCounts() { return &counts; };
  void GetCounts(glm::vec3 cnts) { cnts[0] = counts[0];cnts[1]=counts[1];cnts[2]=counts[2]; return; };
  void SetOrigin(float x, float y, float z) { origin = {x,y,z};};
  void GetDeltas(glm::vec3 &spacing);
  void GetGlobalOrigin(glm::vec3 &origin);
  void SetTransferFunction(TransferFunction* tf) {transfunction = tf;};
  void GetTransferFunction(TransferFunction* tf) {tf = transfunction;};
  void SetSlices(int n, glm::vec4 *s);
  void GetSlices(int &n, glm::vec4 &s);
  void SetIsovalues(int n, float* values);
  void GetIsovalues(int *n, float* values);
protected:
  glm::vec4 *slices;
  glm::vec3 counts;
  glm::vec3 origin;
  glm::vec3 spacing;
  TransferFunction *transfunction;
  int n_slices;
  float *isovalues;
  int n_isovalues;
private:
  DataType type;
  glm::vec3 deltas;
  //unsigned char *uchar_samples;
  float *floatsamples;
  short* shortsamples;
  OSPVolume theOSPVolume;
  OSPData theOSPData;
};
}
}
}
}
#endif
