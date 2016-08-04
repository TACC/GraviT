#include "TransferFunction.h"
#include <iostream>
#include <fstream>

gvt::render::data::primitives::TransferFunction::TransferFunction(){
  n_colors = 0;
  colormap = NULL;
  n_opacities = 0;
  opacitymap = NULL;
  theOSPTransferFunction = NULL;
}
gvt::render::data::primitives::TransferFunction::~TransferFunction() {
}
/*** 
 * update data and commit the device 
 */
bool 
gvt::render::data::primitives::TransferFunction::DeviceCommit() {
  if (theOSPTransferFunction) ospRelease(theOSPTransferFunction);
  theOSPTransferFunction = ospNewTransferFunction("piecewise_linear");

  glm::vec3 color[256];
  int i0 =0, i1 = 1;
  float xmin = colormap[0].x, xmax = colormap[n_colors-1].x;
  for( int i=0;i<256;i++) {
    float x = xmin +(i/(255.0))*(xmax-xmin);
    if(x>xmax) x=xmax;
    while (colormap[i1].x <x)
      i0++,i1++;
    float d = (x - colormap[i0].x) / (colormap[i1].x - colormap[i0].x);
    color[i].x = colormap[i0].y + d * (colormap[i1].y - colormap[i0].y);
    color[i].y = colormap[i0].z + d * (colormap[i1].z - colormap[i0].z);
    color[i].z = colormap[i0].w + d * (colormap[i1].w - colormap[i0].w);
  }
  OSPData oColors = ospNewData(256,OSP_FLOAT3,color);
  ospSetData(theOSPTransferFunction, "colors", oColors);
  float opacity[256];
  i0 = 0, i1=1;
  xmin = opacitymap[0].x, xmax = opacitymap[n_opacities - 1].x;
  for(int i=0;i<256;i++) {
    float x = xmin + (i/(255.0))*(xmax - xmin);
    if(x>xmax) x = xmax;
    while(opacitymap[i1].x < x)
     i0++, i1++;
    float d = (x-opacitymap[i0].x) / (opacitymap[i1].x - opacitymap[i0].x);
    opacity[i] = opacitymap[i0].y + d * (opacitymap[i1].y - opacitymap[i0].y);
  }
  OSPData oAlphas = ospNewData(256,OSP_FLOAT,opacity);
  ospSetData(theOSPTransferFunction, "opacities",oAlphas);
  ospCommit(theOSPTransferFunction);
  return false;
}
/***
 * load a transfer function from a file
 */
void gvt::render::data::primitives::TransferFunction::load(std::string cname, std::string oname) {
  std::ifstream ifs;
  ifs.open(cname.c_str(), std::ios::in);
  if (ifs.fail())
    throw(std::string("error opening colormap file: ") + cname);
  int nc;
  ifs >> nc;
  glm::vec4 *cmap = new glm::vec4[nc];
  glm::vec4 *c = cmap;
  for (int i = 0; i < nc; i++, c++)
    ifs >> c->x >> c->y >> c->z >> c->w;
  ifs.close();

  ifs.open(oname.c_str(), std::ios::in);
  if (ifs.fail())
    throw(std::string("error opening opacity file: ") + oname);
  int no;
  ifs >> no;
  glm::vec2 *omap = new glm::vec2[no];
  glm::vec2 *o = omap;
  for (int i = 0; i < no; i++, o++)
    ifs >> o->x >> o->y;
  ifs.close();
}
