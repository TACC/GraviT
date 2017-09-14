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
PROJECT(GVT_Trace)
SET(GVT_APP_GVT_TRACE OFF CACHE BOOL "Build ConfigFile Backend")

IF (GVT_APP_GVT_TRACE)

    IF(NOT GVT_BE_MANTA)
        MESSAGE("GVT Trace requires Manta Backend")
    ENDIF()

    IF(NOT GVT_FE_CONFFILE)
        MESSAGE("MPI trace requires Config File Front end")
    ENDIF()

    INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR})

    FILE(GLOB_RECURSE GVT_TRACE_SRC ${PROJECT_SOURCE_DIR}/*.cpp ${PROJECT_SOURCE_DIR}/*.C)
    FILE(GLOB_RECURSE GVT_TRACE_HDR ${PROJECT_SOURCE_DIR}/*.h ${PROJECT_SOURCE_DIR}/*.hpp)

    ADD_EXECUTABLE(${PROJECT_NAME} ${GVT_TRACE_SRC} ${GVT_TRACE_HDR})
    TARGET_LINK_LIBRARIES(${PROJECT_NAME} GVT_CORE GVT_RENDER ${MANTA_TARGET_LINK_LIBRARIES} ${MPI_LIBRARIES} )
    CONFIGURE_LIB_MPI()

ENDIF()
