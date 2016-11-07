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
//
// Image.C
//

#include <gvt/render/data/scene/Image.h>

#include <gvt/core/Debug.h>

#include <fstream>
#include <iostream>
#include <mpi.h>
#include <sstream>

using namespace gvt::render::data::scene;

void Image::Write() {

  if (MPI::COMM_WORLD.Get_rank() != 0) return;

  std::string ext;
  switch (format) {
  case PPM:
    ext = ".ppm";
    break;
  default:

    return;
  }

  std::stringstream header;
  header << "P6" << std::endl;
  header << width << " " << height << std::endl;
  header << "255" << std::endl;

  std::fstream file;
  file.open((filename + ext).c_str(), std::fstream::out | std::fstream::trunc | std::fstream::binary);
  file << header.str();

  std::cout << "Image write " << width << " x " << height << std::endl;
  // reverse row order so image is correctly oriented
  for (int j = height - 1; j >= 0; j--) {
    int offset = j * width;
    for (int i = 0; i < width; ++i) {
      int index = 3 * (offset + i);
      // if (rgb[index + 0] == 0 || rgb[index + 1] == 0 || rgb[index + 2] == 0) std::cout << ".";
      file << rgb[index + 0] << rgb[index + 1] << rgb[index + 2];
    }
  }
  std::cout << std::endl;
  file.close();
}
