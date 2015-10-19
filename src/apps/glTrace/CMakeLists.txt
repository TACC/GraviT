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
PROJECT(GLtrace)

MESSAGE(STATUS "---> GLtrace")
IF(NOT GVT_BE_MANTA)
  MESSAGE("GLtrace requires Manta Backend")
ENDIF(NOT GVT_BE_MANTA)

IF(NOT GVT_FE_CONFFILE)
  MESSAGE("GLtrace requires Config File Front end")
ENDIF(NOT GVT_FE_CONFFILE)

FIND_PACKAGE(Manta)

INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR})

SET(FILE_CONFIG_LIB ${MPE_LIB2} ${MPE_LIB1} ${MPI_LIBRARIES} ${Boost_LIBRARIES})

FILE(GLOB_RECURSE GVT_GLTRACE_SRC ${PROJECT_SOURCE_DIR}/*.cpp
  ${PROJECT_SOURCE_DIR}/*.C)
FILE(GLOB_RECURSE GVT_GLTRACE_HDR ${PROJECT_SOURCE_DIR}/*.h
  ${PROJECT_SOURCE_DIR}/*.hpp)

IF(GVT_BE_MANTA)
    INCLUDE_DIRECTORIES(${GVTMANTA_INCLUDE_DIRS})
    SET(FILE_CONFIG_LIB ${FILE_CONFIG_LIB} ${MANTA_TARGET_LINK_LIBRARIES})
    SET(FILE_CONFIG_LIB ${FILE_CONFIG_LIB} "GVTMANTA")
ENDIF(GVT_BE_MANTA)

IF(GVT_BE_OPTIX) 
    INCLUDE_DIRECTORIES(${GVTOPTIX_INCLUDE_DIRS})
    SET(FILE_CONFIG_LIB ${FILE_CONFIG_LIB} "GVTOPTIX" ${GVTOPTIX_LIBRARIES})
ENDIF(GVT_BE_OPTIX)

message("GLtrace file config lib is : ${FILE_CONFIG_LIB}")

include_directories("${GLUT_INSTALL}/include")
find_library(GLUTLIB glut)
if(GLUTLIB)
	set(EXTRA_LIBS ${EXTRA_LIBS} ${GLUTLIB})
endif(GLUTLIB)
find_library(GLULIB GLU)
if(GLULIB)
	set(EXTRA_LIBS ${EXTRA_LIBS} ${GLULIB})
endif(GLULIB)

ADD_EXECUTABLE(${PROJECT_NAME} ${GVT_GLTRACE_SRC} ${GVT_GLTRACE_HDR})
TARGET_LINK_LIBRARIES(${PROJECT_NAME} GVTCORE GVTCONFFILE ${FILE_CONFIG_LIB}
${EXTRA_LIBS}) 

CONFIGURE_LIB_MPI()
