project(GraviT)

cmake_minimum_required(VERSION 2.4)

# version information
set(VERSION_MAJOR 0)
set(VERSION_MINOR 5)
set(VERSION_PATCH 0)


set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(COMMAND cmake_policy)
  # required for mixing libraries with full paths and not...
  cmake_policy(SET CMP0003 NEW)

  # use rpath on MacOS
  cmake_policy(SET CMP0042 NEW)
endif(COMMAND cmake_policy)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the build type: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif(NOT CMAKE_BUILD_TYPE)

# build options
option(BUILD_SHARED_LIBS "Build shared, dynamically-linked libraries" ON)
option(GVT_CORE 		 "Build GraviT core library" ON)
option(GVT_RENDER 		 "Build GraviT geometry rendering library" ON)
option(GVT_VOLUME 		 "Build GraviT volume rendering library" OFF)
option(GVT_ADVECT 		 "Build GraviT particle advection library" OFF)
option(GVT_TIMING 		 "Build GraviT use timing prmimitives" OFF)
option(GVT_TESTING "Create testing and benchmark programs" OFF)

if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
  SET(HOST_NAME $ENV{HOSTNAME})
  message (STATUS "HOSTNAME is ${HOST_NAME}")

  if(HOST_NAME MATCHES "(.)*maverick(.)*")
    set(CMAKE_CXX_FLAGS "-cxxlib=/opt/apps/gcc/4.9.3/ ${CMAKE_CXX_FLAGS}")
    set(CMAKE_C_FLAGS "-cxxlib=/opt/apps/gcc/4.9.3/ ${CMAKE_C_FLAGS}")
    SET(GVT_HOST "Maverick" CACHE STRING "Choose Stampede | Maverick | Other")
  elseif(HOST_NAME MATCHES "login-knl1.stampede(.)*")
    set(CMAKE_CXX_FLAGS "-cxxlib=/opt/apps/gcc/4.9.3/ ${CMAKE_CXX_FLAGS}")
    set(CMAKE_C_FLAGS "-cxxlib=/opt/apps/gcc/4.9.3/ ${CMAKE_C_FLAGS}")
    SET(GVT_HOST "KNL-Stampede" CACHE STRING "Choose Stampede | Maverick | Other")
  elseif(HOST_NAME MATCHES "(.)*stampede(.)*")
    set(CMAKE_CXX_FLAGS "-cxxlib=/opt/apps/gcc/4.9.3/ ${CMAKE_CXX_FLAGS}")
    set(CMAKE_C_FLAGS "-cxxlib=/opt/apps/gcc/4.9.3/ ${CMAKE_C_FLAGS}")
    SET(GVT_HOST "Stampede" CACHE STRING "Choose Stampede | Maverick | Other")
  else()
    SET(GVT_HOST "Other (${HOST_NAME})" CACHE STRING "Choose Stampede | Maverick | Other")
  endif()
endif()



if (GVT_RENDER)
  option(GVT_RENDER_ADAPTER_MANTA  		"Build the SCI Manta ray tracing engine adapter" OFF)
  option(GVT_RENDER_ADAPTER_EMBREE 		"Build the Intel Embree ray tracing engine adapter" ON)
  option(GVT_RENDER_ADAPTER_OPTIX_PRIME 	"Build the NVIDIA Optix Prime ray tracing engine adapter" OFF)
  option(GVT_RENDER_APP 					"Build the GraviT example renderer application" ON)
  option(GVT_GLRENDER_APP			"Build the interactive X application" ON)
  option(GVT_SIMPLE_APP			"Build the simple application" ON)
  option(GVT_SIMPLE_NS_APP			"Build the simple application" ON)
  option(GVT_SIMPLE_FILE_LOAD_APP		"Build the simple application (load obj file)" ON)
  option(GVT_PLY_APP "Build the ply reader application (load ply file)" ON)
  option(GVT_PLY_NS_APP "Build the ply reader application (load ply file)" ON)
  option(GVT_CTEST "Build CTEST Unit and Integration tests" ON)
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


# compiler options

# the compiler is ususally set through environment vars outside the cmake file
# so this is redundant. CMake detects the compiler.
set(GVT_USE_ICC OFF CACHE BOOL "Use Intel Compiler?")
set(GVT_VECTOR_TARGET "AVX" CACHE STRING "Target Vector ISA on host (SSE,AVX,AVX2, AVX512KNL, AVX512SKX)")
SET_PROPERTY(CACHE GVT_VECTOR_TARGET PROPERTY STRINGS NONE SSE AVX AVX2 AVX512KNL AVX512SKX)

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

#TODO best way to just include embre folder ?
set(GVT_THIRDPARTY_EMBREESHADERS "${CMAKE_SOURCE_DIR}/third-party/embree-shaders")


FIND_PACKAGE(IceT REQUIRED)
include_directories(${ICET_INCLUDE_DIRS})
find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

# modules
include(CMake/gravit.cmake)
include(CMake/mpi.cmake)
include(CMake/boost.cmake)
include(CMake/doxygen.cmake)

set(GVT_USE_MPI ON CACHE BOOL "Use MPI") # TODO: should MPI always be required?

IF(${GVT_VECTOR_TARGET} STREQUAL "AVX")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GVT_ARCH_FLAGS__AVX} -DGVT_AVX_TARGET")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${GVT_ARCH_FLAGS__AVX} -DGVT_AVX_TARGET")
ENDIF()

IF(${GVT_VECTOR_TARGET} STREQUAL "AVX2")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GVT_ARCH_FLAGS__AVX2} -DGVT_AVX2_TARGET")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${GVT_ARCH_FLAGS__AVX2} -DGVT_AVX2_TARGET")
ENDIF()

IF(${GVT_VECTOR_TARGET} STREQUAL "AVX512KNL")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GVT_ARCH_FLAGS__AVX512KNL} -DGVT_AVX512KNL_TARGET")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${GVT_ARCH_FLAGS__AVX512KNL} -DGVT_AVX512KNL_TARGET")
ENDIF()

IF(${GVT_VECTOR_TARGET} STREQUAL "AVX512SKX")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GVT_ARCH_FLAGS__AVX512SKX} -DGVT_AVX512SKX_TARGET")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${GVT_ARCH_FLAGS__AVX512SKX} -DGVT_AVX512SKX_TARGET")
ENDIF()

IF(${GVT_VECTOR_TARGET} STREQUAL "SSE")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GVT_ARCH_FLAGS__SSE42} -DGVT_SSE_TARGET")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${GVT_ARCH_FLAGS__SSE42} -DGVT_SSE_TARGET")
ENDIF()

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
  src/gvt/core/context/CoreContext.h
  src/gvt/core/data/Transform.h
  src/gvt/core/context/Database.h
  src/gvt/core/context/DatabaseNode.h
  src/gvt/core/Debug.h
  src/gvt/core/math/RandEngine.h
  src/gvt/core/Math.h
  src/gvt/core/schedule/SchedulerBase.h
  src/gvt/core/Types.h
  src/gvt/core/context/Uuid.h
  src/gvt/core/context/Variant.h

  src/gvt/core/comm/comm.h
  src/gvt/core/comm/communicator.h
  src/gvt/core/comm/communicator/acomm.h
  src/gvt/core/comm/communicator/scomm.h
  src/gvt/core/comm/message.h
  src/gvt/core/comm/vote/vote.h
  src/gvt/core/composite/Composite.h

  src/gvt/core/tracer/tracer.h
)

set(GVT_CORE_SRCS ${GVT_CORE_SRCS}
  src/gvt/core/context/CoreContext.cpp
  src/gvt/core/context/Database.cpp
  src/gvt/core/context/DatabaseNode.cpp
  src/gvt/core/context/Uuid.cpp
  src/gvt/core/context/Variant.cpp

  src/gvt/core/comm/communicator.cpp
  src/gvt/core/comm/communicator/acomm.cpp
  src/gvt/core/comm/communicator/scomm.cpp
  src/gvt/core/comm/message.cpp
  src/gvt/core/comm/vote/vote.cpp

  src/gvt/core/tracer/tracer.cpp
)


set(GVT_RENDER_HDRS ${GVT_RENDER_HDRS}
  src/gvt/render/api/api.h
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
  src/gvt/render/data/primitives/Shade.h
  src/gvt/render/data/primitives/EmbreeMaterial.h
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
  src/gvt/render/composite/composite.h
  src/gvt/render/Schedulers.h
  src/gvt/render/Types.h


  src/gvt/render/composite/IceTComposite.h
  src/gvt/render/composite/ImageComposite.h
  src/gvt/render/tracer/RayTracer.h
  src/gvt/render/tracer/Image/ImageTracer.h
  src/gvt/render/tracer/Domain/DomainTracer.cpp
  src/gvt/render/tracer/Domain/Messages/SendRayList.h
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
  src/gvt/render/composite/composite.cpp

  src/gvt/render/composite/IceTComposite.cpp
  src/gvt/render/composite/ImageComposite.cpp
  src/gvt/render/tracer/RayTracer.cpp
  src/gvt/render/tracer/Image/ImageTracer.cpp
  src/gvt/render/tracer/Domain/DomainTracer.cpp
  src/gvt/render/tracer/Domain/Messages/SendRayList.cpp
)

if (GVT_RENDER_ADAPTER_MANTA)
  add_definitions(-DGVT_RENDER_ADAPTER_MANTA)
  set(GVT_RENDER_HDRS ${GVT_RENDER_HDRS}
    src/gvt/render/adapter/manta/MantaMeshAdapter.h
    src/gvt/render/adapter/manta/DynBVH.h
  )
  set(GVT_RENDER_SRCS ${GVT_RENDER_SRCS} src/gvt/render/adapter/manta/MantaMeshAdapter.cpp)
endif(GVT_RENDER_ADAPTER_MANTA)

if (GVT_RENDER_ADAPTER_OPTIX_PRIME)
  add_definitions(-DGVT_RENDER_ADAPTER_OPTIX)
  set(GVT_RENDER_CUDA_HDRS ${GVT_RENDER_CUDA_HDRS}
    src/gvt/render/adapter/optix/Formats.h
    src/gvt/render/adapter/optix/Transforms.h
    src/gvt/render/adapter/optix/OptixMeshAdapter.h
    src/gvt/render/adapter/optix/Mesh.cuh
    src/gvt/render/adapter/optix/Light.cuh
    src/gvt/render/adapter/optix/Material.cuh
    src/gvt/render/adapter/optix/Ray.cuh
    src/gvt/render/adapter/optix/cutil_math.h
  )

  set(GVT_RENDER_CUDA_SRCS ${GVT_RENDER_CUDA_SRCS}
    src/gvt/render/adapter/optix/OptixMeshAdapter.cu
    src/gvt/render/adapter/optix/Light.cu
    src/gvt/render/adapter/optix/Material.cu
    src/gvt/render/adapter/optix/Ray.cu
  )
  set(GVT_RENDER_SRCS ${GVT_RENDER_SRCS}
    src/gvt/render/adapter/optix/OptixMeshAdapter.cpp
  )
  set(GVT_RENDER_HDRS ${GVT_RENDER_HDRS}
  )
endif(GVT_RENDER_ADAPTER_OPTIX_PRIME)


if (GVT_RENDER_ADAPTER_EMBREE)
  add_definitions(-DGVT_RENDER_ADAPTER_EMBREE)
  set (GVT_RENDER_HDRS ${GVT_RENDER_HDRS}
    src/gvt/render/adapter/embree/EmbreeMeshAdapter.h
  )
  set(GVT_RENDER_SRCS ${GVT_RENDER_SRCS}
    src/gvt/render/adapter/embree/EmbreeMeshAdapter.cpp
  )
endif(GVT_RENDER_ADAPTER_EMBREE)

if (GVT_RENDER_ADAPTER_EMBREE AND GVT_RENDER_ADAPTER_OPTIX_PRIME)
  set (GVT_RENDER_HDRS ${GVT_RENDER_HDRS}
    src/gvt/render/adapter/heterogeneous/HeterogeneousMeshAdapter.h
  )
  set(GVT_RENDER_SRCS ${GVT_RENDER_SRCS}
    src/gvt/render/adapter/heterogeneous/HeterogeneousMeshAdapter.cpp
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
  set(GVT_RENDER_LIBS ${GVT_RENDER_LIBS})

  set(GVT_RENDER_LIBS ${GVT_RENDER_LIBS} tinyobjloader ${ICET_CORE_LIBS} ${ICET_MPI_LIBS})
  include_directories(${TINYOBJPATH})

  # find_package(Boost REQUIRED COMPONENTS system)
  # include_directories(${Boost_INCLUDE_DIRS})
  set(GVT_RENDER_LIBS ${GVT_RENDER_LIBS} ${Boost_LIBRARIES} ${GVT_CORE_LIBS})

  if(GVT_RENDER_ADAPTER_MANTA)
    find_package(Manta REQUIRED)
    set(GVT_RENDER_LIBS ${GVT_RENDER_LIBS} ${MANTA_TARGET_LINK_LIBRARIES})
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
      -gencode arch=compute_50,code=sm_50
    --compiler-options -fPIC )
    #cuda_compile(GVT_RENDER_CUDA_OBJS ${GVT_RENDER_CUDA_HDRS} ${GVT_RENDER_CUDA_SRCS})
    CUDA_ADD_LIBRARY(gvtCudaOptix ${GVT_RENDER_CUDA_SRCS}  OPTIONS ${CUDA_NVCC_FLAGS})

    set(GVT_RENDER_SRCS ${GVT_RENDER_SRCS} ${GVT_RENDER_CUDA_OBJS})
    set(GVT_RENDER_LIBS ${GVT_RENDER_LIBS} gvtCudaOptix)

  endif(GVT_RENDER_ADAPTER_OPTIX_PRIME)

  include_directories(${OptiX_INCLUDE})
  include_directories(${GVT_THIRDPARTY_EMBREESHADERS})

  if (GVT_RENDER_ADAPTER_EMBREE)
    find_package(Embree REQUIRED)
    set(GVT_RENDER_LIBS ${GVT_RENDER_LIBS} ${EMBREE_TARGET_LINK_LIBRARIES})
  endif(GVT_RENDER_ADAPTER_EMBREE)

  add_library(gvtRender STATIC
    ${GVT_CORE_HDRS}
    ${GVT_RENDER_HDRS}
  ${GVT_RENDER_SRCS})

  target_link_libraries(gvtRender ${GVT_RENDER_LIBS})

  install(TARGETS gvtRender
    LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)
endif(GVT_RENDER)


## CTEST
if (GVT_CTEST)

  enable_testing()
  ## Check if run in mpi

  SET(runConfig "" )
  if (GVT_USE_MPI)
    set (no_mpi_proc 2)
    set (runConfig ibrun -np ${no_mpi_proc})
  endif(GVT_USE_MPI)

  ## Enable gvtSimple and gvtFileLoad, needed for tests
  set(GVT_SIMPLE_APP ON CACHE BOOL "Build libfoo shared library" FORCE)
  set(GVT_SIMPLE_FILE_LOAD_APP ON CACHE BOOL "Build libfoo shared library" FORCE)
  set(GVT_BIN_DIR ${PROJECT_BINARY_DIR}/bin)

  ## Testing Configuration
  set(SimpleFileLoadObjLocation ${PROJECT_SOURCE_DIR}/data/geom/bunny.obj)
  set(TestOutputFolder ${PROJECT_SOURCE_DIR}/Test/CTESTtest/testOutput)
  file(MAKE_DIRECTORY ${TestOutputFolder})
  set(ImageDiffTolerance 300)

  ## Integration Test for SimpleApp --Image Scheduler
  add_test( SimpleApp_ImageScheduler  ${runConfig} ${PROJECT_BINARY_DIR}/bin/gvtSimple -image -output ${TestOutputFolder}/test1)
  add_test( SimpleAppImageDiff_ImageScheduler ${PROJECT_BINARY_DIR}/bin/gvtImageDiff -tolerance ${ImageDiffTolerance} -diff "${PROJECT_SOURCE_DIR}/Test/CTESTtest/data/simple.ppm,${TestOutputFolder}/test1.ppm")

  ## Integration Test for SimpleFileLoad --Image Scheduler
  add_test( SimpleFileLoad_ImageScheduler  ${runConfig} ${GVT_BIN_DIR}/gvtFileLoad -image -obj ${SimpleFileLoadObjLocation} -output ${TestOutputFolder}/test2)
  add_test( SimpleFileLoadImageDiff_ImageScheduler ${GVT_BIN_DIR}/gvtImageDiff -tolerance ${ImageDiffTolerance} -diff "${PROJECT_SOURCE_DIR}/Test//CTESTtest/data/bunny.ppm,${TestOutputFolder}/test2.ppm")

  ## Integration Test for SimpleApp --Domain Scheduler
  add_test( SimpleApp_DomainScheduler   ${runConfig} ${GVT_BIN_DIR}/gvtSimple -domain -output ${TestOutputFolder}/test3)
  add_test( SimpleAppImageDiff_DomainScheduler  ${GVT_BIN_DIR}/gvtImageDiff -tolerance ${ImageDiffTolerance} -diff "${PROJECT_SOURCE_DIR}/Test/CTESTtest/data/simple.ppm,${TestOutputFolder}/test3.ppm")

  ## Integration Test for SimpleFileLoad --Domain Scheduler
  add_test( SimpleFileLoad_DomainScheduler ${runConfig} ${GVT_BIN_DIR}/gvtFileLoad -domain -obj ${SimpleFileLoadObjLocation} -output ${TestOutputFolder}/test4)
  add_test( SimpleFileLoadImageDiff_DomainScheduler ${GVT_BIN_DIR}/gvtImageDiff -tolerance ${ImageDiffTolerance} -diff "${PROJECT_SOURCE_DIR}/Test//CTESTtest/data/bunny.ppm,${TestOutputFolder}/test4.ppm")

  ## Image difference program to verify integration tests
  set(GVTIMAGE_DIFF_SRCS  src/apps/render/ImageDiff.cpp)
  add_executable(gvtImageDiff ${GVTIMAGE_DIFF_SRCS})

endif(GVT_CTEST)

if (GVT_SIMPLE_APP)
  # find_package(MPI REQUIRED)
  # include_directories(${MPI_INCLUDE_PATH})
  set(GVTSIMPLE_RENDER_LIBS ${GVTSIMPLE_RENDER_LIBS}  ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES})
  set(GVTSIMPLE_RENDER_SRCS ${GVTSIMPLE_RENDER_SRCS} src/apps/render/SimpleApp.cpp)
  add_executable(gvtSimple ${GVTSIMPLE_RENDER_SRCS})
  target_link_libraries(gvtSimple gvtCore gvtRender ${GVTSIMPLE_RENDER_LIBS} ${GVT_CORE_LIBS})
endif(GVT_SIMPLE_APP)

if (GVT_SIMPLE_NS_APP)
  # find_package(MPI REQUIRED)
  # include_directories(${MPI_INCLUDE_PATH})
  set(GVTSIMPLE_NS_RENDER_LIBS ${GVTSIMPLE_NS_RENDER_LIBS}  ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES})
  set(GVTSIMPLE_NS_RENDER_SRCS ${GVTSIMPLE_NS_RENDER_SRCS} src/apps/ns/SimpleApp.cpp)
  add_executable(gvtSimpleNS ${GVTSIMPLE_NS_RENDER_SRCS})
  target_link_libraries(gvtSimpleNS gvtCore gvtRender ${GVTSIMPLE_NS_RENDER_LIBS} ${GVT_CORE_LIBS})
endif(GVT_SIMPLE_NS_APP)


if(GVT_TESTING)
  find_package(OSPRay)
  include_directories(${OSPRAY_INCLUDE_DIRS} ${PLYPATH})
  set(GVT_TESTING_SRCS ${GVT_TESTING_SRCS} Test/timer.c ${PLYPATH}/ply.c)
  add_executable(osptest ${GVT_TESTING_SRCS} Test/OSPRayTest/BenchOspray.cpp)
  target_link_libraries(osptest ${OSPRAY_LIBRARIES})
  # find_package(MPI REQUIRED)
  # include_directories(${MPI_INCLUDE_PATH})
  add_executable(gvttest ${GVT_TESTING_SRCS} Test/GVTTest/GVTTest.cpp)
  target_link_libraries(gvttest gvtCore gvtRender ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES} ${OSPRAY_LIBRARIES} ${GVT_CORE_LIBS})
endif(GVT_TESTING)

if (GVT_PLY_APP)
  # find_package(MPI REQUIRED)
  # include_directories(${MPI_INCLUDE_PATH}
  include_directories(${PLYPATH})
  set(GVTPLY_RENDER_LIBS ${GVTPLY_RENDER_LIBS}  ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES} ${GVT_CORE_LIBS})
  set(GVTPLY_RENDER_SRCS ${GVTPLY_RENDER_SRCS}
    src/apps/render/PlyApp.cpp
  ${PLYPATH}/ply.c)
  add_executable(gvtPly ${GVTPLY_RENDER_SRCS})
  target_link_libraries(gvtPly gvtCore gvtRender ${GVTPLY_RENDER_LIBS} )
endif(GVT_PLY_APP)

if (GVT_PLY_NS_APP)
  # find_package(MPI REQUIRED)
  # include_directories(${MPI_INCLUDE_PATH} ${PLYPATH})
  include_directories(${PLYPATH})
  set(GVTPLY_NS_RENDER_LIBS ${GVTPLY_NS_RENDER_LIBS}  ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES} ${GVT_CORE_LIBS})
  set(GVTPLY_NS_RENDER_SRCS ${GVTPLY_NS_RENDER_SRCS}
    src/apps/ns/PlyApp.cpp
  ${PLYPATH}/ply.c)
  add_executable(gvtPlyNS ${GVTPLY_NS_RENDER_SRCS})
  set_target_properties (gvtPlyNS PROPERTIES COMPILE_DEFINITIONS "NS")
  target_link_libraries(gvtPlyNS gvtCore gvtRender ${GVTPLY_NS_RENDER_LIBS} )
endif(GVT_PLY_NS_APP)

if (GVT_SIMPLE_FILE_LOAD_APP)
  # find_package(MPI REQUIRED)
  # include_directories(${MPI_INCLUDE_PATH})
  set(GVTSIMPLEFILE_RENDER_LIBS ${GVTSIMPLEFILE_RENDER_LIBS}  ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES})
  set(GVTSIMPLEFILE_RENDER_SRCS ${GVTSIMPLEFILE_RENDER_SRCS} src/apps/render/SimpleFileLoadApp.cpp)
  add_executable(gvtFileLoad ${GVTSIMPLEFILE_RENDER_SRCS})
  target_link_libraries(gvtFileLoad gvtCore gvtRender ${GVTSIMPLEFILE_RENDER_LIBS} )
endif(GVT_SIMPLE_FILE_LOAD_APP)

if (GVT_GLRENDER_APP)
  # find_package(MPI REQUIRED)
  # include_directories(${MPI_INCLUDE_PATH})
  find_package(OpenGL REQUIRED)
  set(GVTGLAPP_RENDER_LIBS ${GVTGLAPP_RENDER_LIBS}  ${MPI_C_LIBRARIES} ${MPI_CXX_LIBRARIES})

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
