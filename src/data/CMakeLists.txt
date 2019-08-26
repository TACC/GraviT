## ============================================================================
## This file is released as part of GraviT2 - scalable, platform independent
## ray tracing tacc.github.io/GraviT2
##
## Copyright 2013-2015 Texas Advanced Computing Center, The University of Texas
## at Austin All rights reserved.
##
## Licensed under the BSD 3-Clause License, (the "License"); you may not use
## this file except in compliance with the License.
## A copy of the License is included with this software in the file LICENSE.
## If your copy does not contain the License, you may obtain a copy of the
## License at:
##
##     http://opensource.org/licenses/BSD-3-Clause
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
## WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under limitations under the License.
## ============================================================================
cmake_minimum_required(VERSION 2.8)
project(gvt2_data)

#include_directories(${gvt2_data_SOURCE_DIR})
include_directories(${PROJECT_SOURCE_DIR})
set(DATA_SOURCES
    Ray.cpp
    )

add_library(${PROJECT_NAME} ${DATA_SOURCES})
#
# data library test code
#
add_executable(${PROJECT_NAME}_tests ${PROJECT_NAME}_tests.cpp)
target_link_libraries(${PROJECT_NAME}_tests ${PROJECT_NAME})
install(TARGETS ${PROJECT_NAME} DESTINATION lib)

install(FILES
    Ray.h
    vec.h
    DESTINATION include/gvt2
    )

