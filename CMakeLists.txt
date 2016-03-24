project(GraviT)

cmake_minimum_required(VERSION 2.4)

# version information
set(VERSION_MAJOR 0)
set(VERSION_MINOR 1)
set(VERSION_PATCH 0)


set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(COMMAND cmake_policy)
  # required for mixing libraries with full paths and not...
  cmake_policy(SET CMP0003 NEW)

  # use rpath on MacOS
  #cmake_policy(SET CMP0042 NEW)
endif(COMMAND cmake_policy)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Choose the build type: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif(NOT CMAKE_BUILD_TYPE)

# build options
option(BUILD_SHARED_LIBS "Build shared, dynamically-linked libraries" ON)
option(GVT_CORE 		 "Build GraviT core library" ON)
option(GVT_RENDER 		 "Build GraviT geometry rendering library" ON)
option(GVT_VOLUME 		 "Build GraviT volume rendering library" OFF)
option(GVT_ADVECT 		 "Build GraviT particle advection library" OFF)
option(GVT_DEBUG 		 "Build GraviT use debug prmimitives" OFF)
option(GVT_TIMING 		 "Build GraviT use timing prmimitives" OFF)
option(GVT_MPE "Use MPE Profileing" OFF)
option(GVT_TESTING "Create testing and benchmark programs" OFF)

if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
  SET(HOST_NAME $ENV{HOSTNAME})
  if(HOST_NAME MATCHES "(.)*maverick(.)*")
    set(CMAKE_CXX_FLAGS "-cxxlib=/opt/apps/gcc/4.9.1/ ${CMAKE_CXX_FLAGS}")
    set(CMAKE_C_FLAGS "-cxxlib=/opt/apps/gcc/4.9.1/ ${CMAKE_C_FLAGS}")
    SET(GVT_HOST "Maverick" CACHE STRING "Choose Stampede | Maverick | Other")
  else()
    if(HOST_NAME MATCHES "(.)*stampede(.)*")
      set(CMAKE_CXX_FLAGS "-cxxlib=/opt/apps/gcc/4.9.1/ ${CMAKE_CXX_FLAGS}")
      set(CMAKE_C_FLAGS "-cxxlib=/opt/apps/gcc/4.9.1/ ${CMAKE_C_FLAGS}")
      SET(GVT_HOST "Stampede" CACHE STRING "Choose Stampede | Maverick | Other")
    else()
      SET(GVT_HOST "Other (${HOST_NAME})" CACHE STRING "Choose Stampede | Maverick | Other")
    endif()
  endif()
endif()



if (GVT_RENDER)
  option(GVT_RENDER_ADAPTER_MANTA  		"Build the SCI Manta ray tracing engine adapter" OFF)
  option(GVT_RENDER_ADAPTER_EMBREE 		"Build the Intel Embree ray tracing engine adapter" ON)
  option(GVT_RENDER_ADAPTER_OPTIX_PRIME 	"Build the NVIDIA Optix Prime ray tracing engine adapter" OFF)
  option(GVT_RENDER_APP 					"Build the GraviT example renderer application" ON)
  option(GVT_GLRENDER_APP			"Build the interactive X application" ON)
  option(GVT_SIMPLE_APP			"Build the simple application" ON)
  option(GVT_SIMPLE_FILE_LOAD_APP		"Build the simple application (load obj file)" ON)
  option(GVT_PLY_APP "Build the ply reader application (load ply file)" ON)
  add_definitions(-DGVT_RENDER)
endif(GVT_RENDER)



set(GVT_BASE_DIR ${PROJECT_SOURCE_DIR})
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/CMake)

set(GVT_BIN_DIR ${PROJECT_BINARY_DIR}/bin)
set(GVT_LIB_DIR ${PROJECT_BINARY_DIR}/lib)
SET(LIBRARY_OUTPUT_PATH ${GVT_LIB_DIR})
SET(EXECUTABLE_OUTPUT_PATH ${GVT_BIN_DIR})

# keep the library path in installed executables
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
if (APPLE)
  set(MACOSX_RPATH ${LIBRARY_OUTPUT_PATH})
endif(APPLE)

#link_directories(${GVT_LIB_DIR})

#MARK_AS_ADVANCED(LIBRARY_OUTPUT_PATH)
#MARK_AS_ADVANCED(EXECUTABLE_OUTPUT_PATH)
#MARK_AS_ADVANCED(CMAKE_INSTALL_PREFIX)
#MARK_AS_ADVANCED(CMAKE_BACKWARDS_COMPATIBILITY)

# compiler options

# the compiler is ususally set through environment vars outside the cmake file
# so this is redundant. CMake detects the compiler.
set(GVT_USE_ICC OFF CACHE BOOL "Use Intel Compiler?")
set(GVT_VECTOR_TARGET "AVX" CACHE STRING "Target Vector ISA on host (SSE,AVX,AVX2)")
# /mark_as_advanced(GVT_USE_ICC)
# mark_as_advanced(GVT_VECTOR_TARGET)

find_package(TBB REQUIRED)
include_directories(${TBB_INCLUDE_DIRS})
link_directories(${TBB_LIBRARY_DIRS})

set(GVT_THIRDPARTY_GLM "third-party/glm")
set(GLM_ROOT_DIR "${PROJECT_SOURCE_DIR}/third-party/glm")
set(GLM_DIR "${PROJECT_SOURCE_DIR}/third-party/glm")
SET(CMAKE_PREFIX_PATH ${PROJECT_SOURCE_DIR}/third-party/glm)
find_package(GLM REQUIRED)
include_directories(${GLM_DIR})
include(CMake/glm.cmake)
#set(GVT_THIRDPARTY_TOBJ "/home/01197/semeraro/Projects/Gravity/gravit/third-party/tinyobjloader-master")
set(GVT_THIRDPARTY_TOBJ "${CMAKE_SOURCE_DIR}/third-party/tinyobjloader-master")

add_subdirectory(${GVT_THIRDPARTY_TOBJ})

find_path(TINYOBJPATH "tiny_obj_loader.h" ${GVT_THIRDPARTY_TOBJ})
if (${TINYOBJPATH} MATCHES "TINYOBJPATH-NOTFOUND")
  MESSAGE(SEND_ERRORS "tiny obj reader not found")
endif (${TINYOBJPATH} MATCHES "TINYOBJPATH-NOTFOUND")

set(GVT_THIRDPARTY_PLY "${CMAKE_SOURCE_DIR}/third-party/ply")
find_path(PLYPATH "ply.h" ${GVT_THIRDPARTY_PLY})
if (${PLYPATH} MATCHES "PLYPATH-NOTFOUND")
  MESSAGE(SEND ERRORS "ply reader header not found")
endif (${PLYPATH} MATCHES "PLYPATH-NOTFOUND")

# if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
#   set(GVT_COMPILER ${CMAKE_CXX_COMPILER_ID} CACHE STRING "GraviT Compiler?")
# elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
#   set(GVT_COMPILER ${CMAKE_CXX_COMPILER_ID} CACHE STRING "GraviT Compiler?")
# elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
#   #set(CMAKE_CXX_FLAGS "-Wall  -Wno-reorder -Wno-unused-variable")
#   set(GVT_COMPILER ${CMAKE_CXX_COMPILER_ID} CACHE STRING "GraviT Compiler?")
# set(GVT_USE_ICC ON CACHE BOOL "Use Intel Compiler?")
# elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
#   set(GVT_COMPILER ${CMAKE_CXX_COMPILER_ID} CACHE STRING "GraviT Compiler?")
# endif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")


# modules
include(CMake/gravit.cmake)
include(CMake/mpi.cmake)
include(CMake/boost.cmake)
include(CMake/doxygen.cmake)

set(GVT_USE_MPI ON CACHE BOOL "Use MPI") # TODO: should MPI always be required?
if(GVT_MPE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DGVT_USE_MPE=1")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DGVT_USE_MPE=1")
endif(GVT_MPE)

IF(${GVT_VECTOR_TARGET} STREQUAL "AVX")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GVT_ARCH_FLAGS__AVX} -DGVT_AVX_TARGET")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${GVT_ARCH_FLAGS__AVX} -DGVT_AVX_TARGET")
ENDIF()

IF(${GVT_VECTOR_TARGET} STREQUAL "AVX2")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GVT_ARCH_FLAGS__AVX2} -DGVT_AVX2_TARGET")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${GVT_ARCH_FLAGS__AVX2} -DGVT_AVX2_TARGET")
ENDIF()

IF(${GVT_VECTOR_TARGET} STREQUAL "SSE")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GVT_ARCH_FLAGS__SSE42} -DGVT_SSE_TARGET")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${GVT_ARCH_FLAGS__SSE42} -DGVT_SSE_TARGET")
ENDIF()

if(GVT_DEBUG)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DGVT_USE_DEBUG=1")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DGVT_USE_DEBUG=1")
endif(GVT_DEBUG)

if(GVT_TIMING)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DGVT_USE_TIMING=1")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DGVT_USE_TIMING=1")
endif(GVT_TIMING)

# show all warnings, but disable c11 warnings
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-reorder -Wno-unused-variable")
# use C++11 foo explicitly
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
IF(APPLE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
ENDIF(APPLE)

set(GVT_SRC_DIR ${PROJECT_SOURCE_DIR}/src)
include_directories(${GVT_SRC_DIR})


set(GVT_CORE_HDRS ${GVT_CORE_HDRS}
  src/gvt/core/Actor.h
  src/gvt/core/Adapter.h
  src/gvt/core/Algorithm.h
  src/gvt/core/CoreContext.h
  src/gvt/core/data/Transform.h
  src/gvt/core/Data.h
  src/gvt/core/Database.h
  src/gvt/core/DatabaseNode.h
  src/gvt/core/Debug.h
  src/gvt/core/math/RandEngine.h
  src/gvt/core/Math.h
  src/gvt/core/mpi/Comm.h
  src/gvt/core/mpi/CommData.h
  src/gvt/core/mpi/SingleNode.h
  src/gvt/core/mpi/Wrapper.h
  src/gvt/core/schedule/SchedulerBase.h
  src/gvt/core/Schedule.h
  src/gvt/core/String.h
  src/gvt/core/Types.h
  src/gvt/core/Uuid.h
  src/gvt/core/Variant.h
)

set(GVT_CORE_SRCS ${GVT_CORE_SRCS}
  src/gvt/core/CoreContext.cpp
  src/gvt/core/Database.cpp
  src/gvt/core/DatabaseNode.cpp
  src/gvt/core/Uuid.cpp
  src/gvt/core/Variant.cpp
)


set(GVT_RENDER_HDRS ${GVT_RENDER_HDRS}
  src/gvt/render/actor/Ray.h
  src/gvt/render/algorithm/DomainTracer.h
  src/gvt/render/algorithm/HybridTracer.h
  src/gvt/render/algorithm/ImageTracer.h
  src/gvt/render/algorithm/TracerBase.h
  src/gvt/render/algorithm/Tracers.h
  src/gvt/render/RenderContext.h
  src/gvt/render/data/DerivedTypes.h
  src/gvt/render/data/reader/ObjReader.h
  src/gvt/render/data/reader/PlyReader.h
  src/gvt/render/data/Domains.h
  src/gvt/render/data/Primitives.h
  src/gvt/render/data/primitives/BBox.h
  src/gvt/render/data/primitives/Material.h
  src/gvt/render/data/primitives/Mesh.h
  src/gvt/render/data/scene/gvtCamera.h
  src/gvt/render/data/scene/CameraConfig.h
  src/gvt/render/data/scene/ColorAccumulator.h
  src/gvt/render/data/scene/Image.h
  src/gvt/render/data/scene/Light.h
  src/gvt/render/data/accel/AbstractAccel.h
  src/gvt/render/data/accel/BVH.h
  src/gvt/render/schedule/DomainScheduler.h
  src/gvt/render/schedule/hybrid/AdaptiveSendSchedule.h
  src/gvt/render/schedule/hybrid/GreedySchedule.h
  src/gvt/render/schedule/hybrid/HybridScheduleBase.h
  src/gvt/render/schedule/hybrid/LoadAnotherSchedule.h
  src/gvt/render/schedule/hybrid/LoadAnyOnceSchedule.h
  src/gvt/render/schedule/hybrid/LoadManySchedule.h
  src/gvt/render/schedule/hybrid/LoadOnceSchedule.h
  src/gvt/render/schedule/hybrid/RayWeightedSpreadSchedule.h
  src/gvt/render/schedule/hybrid/SpreadSchedule.h
  src/gvt/render/schedule/HybridScheduler.h
  src/gvt/render/schedule/ImageScheduler.h
  src/gvt/render/Schedulers.h
  src/gvt/render/Types.h
)


set(GVT_RENDER_SRCS ${GVT_RENDER_SRCS}
  src/gvt/render/actor/Ray.cpp
  src/gvt/render/RenderContext.cpp
  src/gvt/render/data/reader/ObjReader.cpp
  src/gvt/render/data/reader/PlyReader.cpp
  src/gvt/render/data/primitives/BBox.cpp
  src/gvt/render/data/primitives/Material.cpp
  src/gvt/render/data/primitives/Mesh.cpp
  src/gvt/render/data/scene/gvtCamera.cpp
  src/gvt/render/data/scene/ColorAccumulator.cpp
  src/gvt/render/data/scene/Image.cpp
  src/gvt/render/data/scene/Light.cpp
  src/gvt/render/data/accel/BVH.cpp
)


if (GVT_RENDER_ADAPTER_MANTA)
  add_definitions(-DGVT_RENDER_ADAPTER_MANTA)
	set(GVT_RENDER_HDRS ${GVT_RENDER_HDRS}
		src/gvt/render/adapter/manta/data/MantaMeshAdapter.h
		src/gvt/render/adapter/manta/data/Transforms.h
		src/gvt/render/adapter/manta/override/DynBVH.h
		src/gvt/render/adapter/manta/Wrapper.h
		src/gvt/render/adapter/optix/data/Mesh.cuh
		src/gvt/render/adapter/optix/data/Light.cuh
		src/gvt/render/adapter/optix/data/Material.cuh
		src/gvt/render/adapter/optix/data/Ray.cuh
		src/gvt/render/adapter/optix/data/cutil_math.h
		)

  set(GVT_RENDER_SRCS ${GVT_RENDER_SRCS}
    src/gvt/render/adapter/manta/data/MantaMeshAdapter.cpp
  )

endif(GVT_RENDER_ADAPTER_MANTA)


if (GVT_RENDER_ADAPTER_OPTIX_PRIME)
  add_definitions(-DGVT_RENDER_ADAPTER_OPTIX)
	set(GVT_RENDER_CUDA_HDRS ${GVT_RENDER_CUDA_HDRS}
		src/gvt/render/adapter/optix/data/Formats.h
		src/gvt/render/adapter/optix/data/Transforms.h
		src/gvt/render/adapter/optix/data/OptixMeshAdapter.h
		src/gvt/render/adapter/optix/Wrapper.h
		)

	set(GVT_RENDER_CUDA_SRCS ${GVT_RENDER_CUDA_SRCS}
	        src/gvt/render/adapter/optix/data/OptixMeshAdapter.cu
		src/gvt/render/adapter/optix/data/Light.cu
		src/gvt/render/adapter/optix/data/Material.cu
		src/gvt/render/adapter/optix/data/Ray.cu
		)
	set(GVT_RENDER_SRCS ${GVT_RENDER_SRCS}
		src/gvt/render/adapter/optix/data/OptixMeshAdapter.cpp
		)
	set(GVT_RENDER_HDRS ${GVT_RENDER_HDRS}
		)
endif(GVT_RENDER_ADAPTER_OPTIX_PRIME)


if (GVT_RENDER_ADAPTER_EMBREE)
  add_definitions(-DGVT_RENDER_ADAPTER_EMBREE)
  set (GVT_RENDER_HDRS ${GVT_RENDER_HDRS}
    src/gvt/render/adapter/embree/data/EmbreeMeshAdapter.h
    src/gvt/render/adapter/embree/Wrapper.h
  )
  set(GVT_RENDER_SRCS ${GVT_RENDER_SRCS}
    src/gvt/render/adapter/embree/data/EmbreeMeshAdapter.cpp
  )
endif(GVT_RENDER_ADAPTER_EMBREE)

if (GVT_RENDER_ADAPTER_EMBREE AND GVT_RENDER_ADAPTER_OPTIX_PRIME)
  set (GVT_RENDER_HDRS ${GVT_RENDER_HDRS}
    src/gvt/render/adapter/heterogeneous/data/HeterogeneousMeshAdapter.h
    src/gvt/render/adapter/heterogeneous/Wrapper.h
  )
  set(GVT_RENDER_SRCS ${GVT_RENDER_SRCS}
    src/gvt/render/adapter/heterogeneous/data/HeterogeneousMeshAdapter.cpp
  )
endif(GVT_RENDER_ADAPTER_EMBREE AND GVT_RENDER_ADAPTER_OPTIX_PRIME)


if (GVT_CORE)
  find_package(Boost REQUIRED COMPONENTS system)
  include_directories(${Boost_INCLUDE_DIRS})

  set(GVT_CORE_LIBS ${LIBS} ${Boost_LIBRARIES} ${TBB_LIBRARIES})
  message(STATUS ${TBB_LIBRARIES})
  add_library(gvtCore STATIC
    ${GVT_CORE_HDRS}
    ${GVT_CORE_SRCS}
  )

  target_link_libraries(gvtCore ${GVT_CORE_LIBS})

  install(TARGETS gvtCore
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
  )
endif(GVT_CORE)


if (GVT_RENDER)
  # TODO: PlyReader in gvtRender requires Manta,
  #       should only need Manta if including Manta adapter
  # find_package(Manta REQUIRED)
  set(GVT_RENDER_LIBS ${GVT_RENDER_LIBS})
  set(GVT_RENDER_LIBS ${GVT_RENDER_LIBS} tinyobjloader)
  include_directories(${TINYOBJPATH})

    find_package(Boost REQUIRED COMPONENTS thread timer system)
    include_directories(${Boost_INCLUDE_DIRS})
    set(GVT_RENDER_LIBS ${GVT_RENDER_LIBS} ${Boost_LIBRARIES} ${GVT_CORE_LIBS})

  if(GVT_RENDER_ADAPTER_MANTA)
    set(GVT_RENDER_LIBS ${GVT_RENDER_LIBS } ${MANTA_TARGET_LINK_LIBRARIES})
  endif(GVT_RENDER_ADAPTER_MANTA)

    if (GVT_RENDER_ADAPTER_OPTIX_PRIME)
    	find_package(CUDA REQUIRED)
    	find_package(OptiX REQUIRED)

	set(CUDA_SEPARABLE_COMPILATION ON)
	set(CUDA_PROPAGATE_HOST_FLAGS OFF)

    	include_directories(${CUDA_INCLUDE_DIRS})
    	include_directories(${OptiX_INCLUDE})
	set(OptiX_LIBRARIES ${optix_LIBRARY} ${optix_prime_LIBRARY})
    	set(GVT_RENDER_LIBS ${GVT_RENDER_LIBS} ${CUDA_LIBRARIES} ${OptiX_LIBRARIES})
    	set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
    		--compiler-options -fno-strict-aliasing -lineinfo -use_fast_math
                -gencode arch=compute_20,code=sm_20
    		-gencode arch=compute_30,code=sm_30
    		-gencode arch=compute_35,code=sm_35
                --compiler-options '-fPIC'
    		)
    	#cuda_compile(GVT_RENDER_CUDA_OBJS ${GVT_RENDER_CUDA_HDRS} ${GVT_RENDER_CUDA_SRCS})
	CUDA_ADD_LIBRARY(gvtCudaOptix ${GVT_RENDER_CUDA_SRCS}  OPTIONS "${CUDA_NVCC_FLAGS}")

    	set(GVT_RENDER_SRCS ${GVT_RENDER_SRCS}
    		${GVT_RENDER_CUDA_OBJS}
    		)

	set(GVT_RENDER_LIBS ${GVT_RENDER_LIBS} gvtCudaOptix)

    endif(GVT_RENDER_ADAPTER_OPTIX_PRIME)

	include_directories(${OptiX_INCLUDE})

	if (GVT_RENDER_ADAPTER_EMBREE)
		find_package(Embree REQUIRED)
		set(GVT_RENDER_LIBS ${GVT_RENDER_LIBS} ${EMBREE_TARGET_LINK_LIBRARIES})
	endif(GVT_RENDER_ADAPTER_EMBREE)

	add_library(gvtRender STATIC
		${GVT_CORE_HDRS}
		${GVT_CORE_SRCS}
		${GVT_RENDER_HDRS}
		${GVT_RENDER_SRCS}
		)

	target_link_libraries(gvtRender ${GVT_RENDER_LIBS})

	install(TARGETS gvtRender
		LIBRARY DESTINATION lib
		ARCHIVE DESTINATION lib
		)
endif(GVT_RENDER)

if (GVT_SIMPLE_APP)
  if (GVT_MPE)
    if (DEFINED ENV{MPEHOME})
      set(MPE_HOME_DIR $ENV{MPEHOME})
      set (MPE_INCLUDE_DIR ${MPE_HOME_DIR}/include)
      include_directories(${MPE_INCLUDE_DIR})
      FIND_LIBRARY(MPE_LIB1 NAME mpe PATHS ${MPE_HOME_DIR} PATH_SUFFIXES lib )
      FIND_LIBRARY(MPE_LIB2 NAME lmpe PATHS ${MPE_HOME_DIR} PATH_SUFFIXES lib)
      set(GVTSIMPLE_RENDER_LIBS ${GVTSIMPLE_RENDER_LIBS} ${MPE_LIB2})
      set(GVTSIMPLE_RENDER_LIBS ${GVTSIMPLE_RENDER_LIBS} ${MPE_LIB1})
    endif(DEFINED ENV{MPEHOME})
  endif (GVT_MPE)
  find_package(MPI REQUIRED)
  include_directories(${MPI_INCLUDE_PATH})
  set(GVTSIMPLE_RENDER_LIBS ${GVTSIMPLE_RENDER_LIBS}  ${MPI_LIBRARIES})
  set(GVTSIMPLE_RENDER_SRCS ${GVTSIMPLE_RENDER_SRCS} src/apps/render/SimpleApp.cpp)
  add_executable(gvtSimple ${GVTSIMPLE_RENDER_SRCS})
  target_link_libraries(gvtSimple gvtCore gvtRender ${GVTSIMPLE_RENDER_LIBS} ${GVT_CORE_LIBS})
endif(GVT_SIMPLE_APP)

if(GVT_TESTING)
  find_package(OSPRay)
  include_directories(${OSPRAY_INCLUDE_DIRS} ${PLYPATH})
  set(GVT_TESTING_SRCS ${GVT_TESTING_SRCS} Test/timer.c ${PLYPATH}/ply.c)
  add_executable(osptest ${GVT_TESTING_SRCS} Test/OSPRayTest/BenchOspray.cpp)
  target_link_libraries(osptest ${OSPRAY_LIBRARIES})
  find_package(MPI REQUIRED)
  include_directories(${MPI_INCLUDE_PATH})
  add_executable(gvttest ${GVT_TESTING_SRCS} Test/GVTTest/GVTTest.cpp)
  target_link_libraries(gvttest gvtCore gvtRender ${MPI_LIBRARIES} ${OSPRAY_LIBRARIES} ${GVT_CORE_LIBS})
endif(GVT_TESTING)

if (GVT_PLY_APP)
  if (GVT_MPE)
    if (DEFINED ENV{MPEHOME})
      set(MPE_HOME_DIR $ENV{MPEHOME})
      set (MPE_INCLUDE_DIR ${MPE_HOME_DIR}/include)
      include_directories(${MPE_INCLUDE_DIR})
      FIND_LIBRARY(MPE_LIB1 NAME mpe PATHS ${MPE_HOME_DIR} PATH_SUFFIXES lib )
      FIND_LIBRARY(MPE_LIB2 NAME lmpe PATHS ${MPE_HOME_DIR} PATH_SUFFIXES lib)
      set(GVTPLY_RENDER_LIBS ${GVTPLY_RENDER_LIBS} ${MPE_LIB2})
      set(GVTPLY_RENDER_LIBS ${GVTPLY_RENDER_LIBS} ${MPE_LIB1})
    endif(DEFINED ENV{MPEHOME})
  endif (GVT_MPE)
  find_package(MPI REQUIRED)
  include_directories(${MPI_INCLUDE_PATH} ${PLYPATH})
  set(GVTPLY_RENDER_LIBS ${GVTPLY_RENDER_LIBS}  ${MPI_LIBRARIES} ${GVT_CORE_LIBS})
  set(GVTPLY_RENDER_SRCS ${GVTPLY_RENDER_SRCS}
    src/apps/render/PlyApp.cpp
  ${PLYPATH}/ply.c)
  add_executable(gvtPly ${GVTPLY_RENDER_SRCS})
  target_link_libraries(gvtPly gvtCore gvtRender ${GVTPLY_RENDER_LIBS} )
endif(GVT_PLY_APP)

if (GVT_SIMPLE_FILE_LOAD_APP)
  if (GVT_MPE)
    if (DEFINED ENV{MPEHOME})
      set(MPE_HOME_DIR $ENV{MPEHOME})
      set (MPE_INCLUDE_DIR ${MPE_HOME_DIR}/include)
      include_directories(${MPE_INCLUDE_DIR})
      FIND_LIBRARY(MPE_LIB1 NAME mpe PATHS ${MPE_HOME_DIR} PATH_SUFFIXES lib )
      FIND_LIBRARY(MPE_LIB2 NAME lmpe PATHS ${MPE_HOME_DIR} PATH_SUFFIXES lib)
      set(GVTSIMPLEFILE_RENDER_LIBS ${GVTSIMPLEFILE_RENDER_LIBS} ${MPE_LIB2} ${GVT_CORE_LIBS})
      set(GVTSIMPLEFILE_RENDER_LIBS ${GVTSIMPLEFILE_RENDER_LIBS} ${MPE_LIB1})
    endif(DEFINED ENV{MPEHOME})
  endif (GVT_MPE)
  find_package(MPI REQUIRED)
  include_directories(${MPI_INCLUDE_PATH})
  set(GVTSIMPLEFILE_RENDER_LIBS ${GVTSIMPLEFILE_RENDER_LIBS}  ${MPI_LIBRARIES})
  set(GVTSIMPLEFILE_RENDER_SRCS ${GVTSIMPLEFILE_RENDER_SRCS} src/apps/render/SimpleFileLoadApp.cpp)
  add_executable(gvtFileLoad ${GVTSIMPLEFILE_RENDER_SRCS})
  target_link_libraries(gvtFileLoad gvtCore gvtRender ${GVTSIMPLEFILE_RENDER_LIBS} )
endif(GVT_SIMPLE_FILE_LOAD_APP)

if (GVT_GLRENDER_APP)
  find_package(MPI REQUIRED)
  find_package(OpenGL REQUIRED)
  include_directories(${MPI_INCLUDE_PATH})
  set(GVTGLAPP_RENDER_LIBS ${GVTGLAPP_RENDER_LIBS}  ${MPI_LIBRARIES})

  find_library(GLUTLIB glut)
  if(GLUTLIB)
    set(GVTGLAPP_RENDER_LIBS ${GVTGLAPP_RENDER_LIBS} ${OpenGL_LIBRARIES} ${GLUTLIB} ${GVT_CORE_LIBS})
  endif(GLUTLIB)
  find_library(GLULIB GLU)
  if(GLULIB)
    set(GVTGLAPP_RENDER_LIBS ${GVTGLAPP_RENDER_LIBS} ${GLULIB})
  endif(GLULIB)

  set(GVTGLAPP_RENDER_HDRS ${GVTGLAPP_RENDER_HDRS}
    src/apps/render/ConfigFileLoader.h
  )

  set(GVTGLAPP_RENDER_SRCS ${GVTGLAPP_RENDER_SRCS}
    src/apps/render/ConfigFileLoader.cpp
    src/apps/render/GLTrace.cpp
    ${PLYPATH}/ply.c
  )

  add_executable(glTracer
    ${GVTGLAPP_RENDER_HDRS}
    ${GVTGLAPP_RENDER_SRCS}
  )

  target_link_libraries(glTracer ${OPENGL_gl_LIBRARY} ${GVTGLAPP_RENDER_LIBS} gvtCore gvtRender)

  install(TARGETS glTracer
    RUNTIME DESTINATION bin
  )
endif(GVT_GLRENDER_APP)
