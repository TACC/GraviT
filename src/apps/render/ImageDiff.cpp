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
#include <assert.h>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <vector>
using namespace std;

#include "ParseCommandLine.h"

string trim(string &str) {
  size_t first = str.find_first_not_of(' ');
  size_t last = str.find_last_not_of(' ');
  return str.substr(first, (last - first + 1));
}

bool is_file_exist(const std::string fileName) {
  std::ifstream infile(fileName.c_str());
  return infile.good();
}

unsigned int Difference(std::string fileA, std::string fileB) {
  std::ifstream fA(fileA.c_str());
  std::ifstream fB(fileB.c_str());

  char Ac;
  char Bc;

  unsigned int difference = 0;

  while (fA.get(Ac)) // loop getting single characters
  {
    fB.get(Bc);
    unsigned int k = (unsigned int)Ac;
    unsigned int kk = (unsigned int)Bc;
    unsigned int diff = (k > kk) ? k - kk : kk - k;
    difference += diff;
  }

  fA.close();
  fB.close();

  return difference;
}

int main(int argc, char **argv) {
  ParseCommandLine cmd("gvtImageDiff");
  cmd.addoption("diff", ParseCommandLine::PATH, "2 args, Image A & Image B", 2);
  cmd.addoption("tolerance", ParseCommandLine::INT, "The tolerance of the image.", 1);
  cmd.parse(argc, argv);

  int tolerance = 10;

  if (!cmd.isSet("diff")) {
    std::cout << "No Images to diff." << std::endl;
    return -1;
  }

  if (cmd.isSet("tolerance")) {
    tolerance = cmd.getValue<int>("tolerance")[0];
  }

  std::cout << "Setting tolerance to: " << tolerance << std::endl;

  std::string ImageA = trim(cmd.getValue<std::string>("diff")[0]);
  std::string ImageB = trim(cmd.getValue<std::string>("diff")[1]);

  std::cout << "Comparing: " << ImageA << " With : " << ImageB << std::endl;

  assert(is_file_exist(ImageA) == true && "File A does not exist");
  assert(is_file_exist(ImageB) == true && "File B does not exist");

  unsigned int difference = Difference(ImageA, ImageB);

  std::cout << "Image Difference is: " << difference << std::endl;

  return difference < tolerance ? 0 : -1;
}
