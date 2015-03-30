project(GraviT)

cmake_minimum_required(VERSION 2.4)

# version information
set(VERSION_MAJOR 0)
set(VERSION_MINOR 1)
set(VERSION_PATCH 0)


if(COMMAND cmake_policy)
	# required for mixing libraries with full paths and not...
    cmake_policy(SET CMP0003 NEW)

    # use rpath on MacOS
    cmake_policy(SET CMP0042 NEW)
endif(COMMAND cmake_policy)

# show all warnings, but disable c11 warnings
set(CMAKE_CXX_FLAGS "-Wall -Wno-c++11-extensions -Wno-reorder -Wno-unused-variable")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Choose the build type: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif(NOT CMAKE_BUILD_TYPE)

# build options
option(BUILD_SHARED_LIBS "Build shared, dynamically-linked libraries" ON)
option(GVT_CORE 		 "Build GraviT core library" ON)
option(GVT_RENDER 		 "Build GraviT geometry rendering library" ON)
option(GVT_VOLUME 		 "Build GraviT volume rendering library" OFF)
option(GVT_ADVECT 		 "Build GraviT particle advection library" OFF)


if (GVT_RENDER)
	option(GVT_RENDER_ADAPTER_MANTA  		"Build the SCI Manta ray tracing engine adapter" ON)
	option(GVT_RENDER_ADAPTER_EMBREE 		"Build the Intel Embree ray tracing engine adapter" OFF)
	option(GVT_RENDER_ADAPTER_OPTIX_PRIME 	"Build the NVIDIA Optix Prime ray tracing engine adapter" OFF)
	option(GVT_RENDER_APP 					"Build the GraviT example renderer application" ON)
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
set(GVT_USE_MPI ON CACHE BOOL "Use MPI") # TODO: should MPI always be required?
set(GVT_USE_ICC OFF CACHE BOOL "Use Intel Compiler?")
set(GVT_VECTOR_TARGET "AVX" CACHE STRING "Target Vector ISA on host (SSE,AVX,AVX2)")
mark_as_advanced(GVT_USE_ICC)
mark_as_advanced(GVT_VECTOR_TARGET)

set(GVT_THIRDPARTY_GLM "third-party/glm")
find_package(GLM REQUIRED)

if (GLM_FOUND)
    include_directories(${GLM_INCLUDE_DIRS})
endif(GLM_FOUND)

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  set(GVT_COMPILER ${CMAKE_CXX_COMPILER_ID} CACHE STRING "GraviT Compiler?")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  set(GVT_COMPILER ${CMAKE_CXX_COMPILER_ID} CACHE STRING "GraviT Compiler?")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
  set(GVT_COMPILER ${CMAKE_CXX_COMPILER_ID} CACHE STRING "GraviT Compiler?")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  set(GVT_COMPILER ${CMAKE_CXX_COMPILER_ID} CACHE STRING "GraviT Compiler?")
endif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")


# modules
include(CMake/gravit.cmake)
include(CMake/mpi.cmake)
include(CMake/boost.cmake)
include(CMake/doxygen.cmake)

set(GVT_SRC_DIR ${PROJECT_SOURCE_DIR}/src)
include_directories(${GVT_SRC_DIR})


set(GVT_CORE_HDRS ${GVT_CORE_HDRS}
	src/gvt/core/Actor.h
	src/gvt/core/Adapter.h
	src/gvt/core/Algorithm.h
	src/gvt/core/Context.h
	src/gvt/core/data/Transform.h
	src/gvt/core/Data.h
	src/gvt/core/Database.h
	src/gvt/core/DatabaseNode.h
	src/gvt/core/Debug.h
	src/gvt/core/math/Angle.h
	src/gvt/core/math/Constants.h
	src/gvt/core/math/Function.h
	src/gvt/core/math/Matrix.h
	src/gvt/core/math/MatrixOperation.inl
	src/gvt/core/math/Unit.h
	src/gvt/core/math/Vector.h
	src/gvt/core/math/VectorOperators.inl
	src/gvt/core/Math.h
	src/gvt/core/mpi/Comm.h
	src/gvt/core/mpi/CommData.h
	src/gvt/core/mpi/SingleNode.h
	src/gvt/core/mpi/Wrapper.h
	src/gvt/core/schedule/SchedulerBase.h
	src/gvt/core/schedule/TaskScheduling.h
	src/gvt/core/schedule/ThreadSafeQueue.h
	src/gvt/core/Schedule.h
	src/gvt/core/Types.h
	)


set(GVT_CORE_SRCS ${GVT_CORE_SRCS}
	src/gvt/core/Context.cpp
	src/gvt/core/Database.cpp
	src/gvt/core/DatabaseNode.cpp
	src/gvt/core/schedule/TaskScheduling.cpp
	)


set(GVT_RENDER_HDRS ${GVT_RENDER_HDRS}
	src/gvt/render/actor/Ray.h
	src/gvt/render/algorithm/AdaptParam.h
	src/gvt/render/algorithm/DomainTracer.h
	src/gvt/render/algorithm/HybridTracer.h
	src/gvt/render/algorithm/ImageTracer.h
	src/gvt/render/algorithm/MetaProcessQueue.h
	src/gvt/render/algorithm/TracerBase.h
	src/gvt/render/algorithm/Tracers.h
	src/gvt/render/Attributes.h
	src/gvt/render/data/Dataset.h
	src/gvt/render/data/DerivedTypes.h
	src/gvt/render/data/domain/AbstractDomain.h
	src/gvt/render/data/domain/Cell.h
	src/gvt/render/data/domain/GeometryDomain.h
	src/gvt/render/data/domain/reader/ObjReader.h
	src/gvt/render/data/domain/reader/PlyReader.h
	src/gvt/render/data/domain/VolumeDomain.h
	src/gvt/render/data/Domains.h
	src/gvt/render/data/Primitives.h
	src/gvt/render/data/primitives/BBox.h
	src/gvt/render/data/primitives/Material.h
	src/gvt/render/data/primitives/Mesh.h
	src/gvt/render/data/scene/Camera.h
	src/gvt/render/data/scene/CameraConfig.h
	src/gvt/render/data/scene/ColorAccumulator.h
	src/gvt/render/data/scene/Image.h
	src/gvt/render/data/scene/Light.h
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
	)


set(GVT_RENDER_SRCS ${GVT_RENDER_SRCS}
	src/gvt/render/actor/Ray.cpp
	src/gvt/render/Attributes.cpp
	src/gvt/render/data/Dataset.cpp
	src/gvt/render/data/domain/AbstractDomain.cpp
	src/gvt/render/data/domain/Cell.cpp
	src/gvt/render/data/domain/GeometryDomain.cpp
	src/gvt/render/data/domain/reader/ObjReader.cpp
	src/gvt/render/data/domain/reader/PlyReader.cpp
	src/gvt/render/data/domain/VolumeDomain.cpp
	src/gvt/render/data/primitives/BBox.cpp
	src/gvt/render/data/primitives/Material.cpp
	src/gvt/render/data/primitives/Mesh.cpp
	src/gvt/render/data/scene/Camera.cpp
	src/gvt/render/data/scene/ColorAccumulator.cpp
	src/gvt/render/data/scene/Image.cpp
	src/gvt/render/data/scene/Light.cpp
	)


if (GVT_RENDER_ADAPTER_MANTA)
	set(GVT_RENDER_HDRS ${GVT_RENDER_HDRS}
		src/gvt/render/adapter/manta/data/domain/MantaDomain.h
		src/gvt/render/adapter/manta/data/Transforms.h
		src/gvt/render/adapter/manta/override/DynBVH.h
		src/gvt/render/adapter/manta/Wrapper.h
		)

	set(GVT_RENDER_SRCS ${GVT_RENDER_SRCS}
		src/gvt/render/adapter/manta/algorithm/MantaProcessQueue.cpp
		src/gvt/render/adapter/manta/data/domain/MantaDomain.cpp
		)
endif(GVT_RENDER_ADAPTER_MANTA)


if (GVT_RENDER_ADAPTER_OPTIX_PRIME)
	set(GVT_RENDER_CUDA_HDRS ${GVT_RENDER_CUDA_HDRS}
		src/gvt/render/adapter/optix/data/domain/OptixDomain.h
		src/gvt/render/adapter/optix/data/Formats.h
		src/gvt/render/adapter/optix/data/Transforms.h
		src/gvt/render/adapter/optix/Wrapper.h
		)

	set(GVT_RENDER_CUDA_SRCS ${GVT_RENDER_CUDA_SRCS}
		src/gvt/render/adapter/optix/data/domain/OptixDomain.cu
		src/gvt/render/adapter/optix/data/Transforms.cu
		)
endif(GVT_RENDER_ADAPTER_OPTIX_PRIME)


if (GVT_RENDER_ADAPTER_EMBREE)
	set (GVT_RENDER_HDRS ${GVT_RENDER_HDRS}
		src/gvt/render/adapter/embree/data/Transforms.h
		src/gvt/render/adapter/embree/Wrapper.h
		)
endif(GVT_RENDER_ADAPTER_EMBREE)


if (GVT_CORE)
    find_package(Boost REQUIRED COMPONENTS system)
    include_directories(${Boost_INCLUDE_DIRS})
    set(GVT_CORE_LIBS ${LIBS} ${Boost_LIBRARIES})

	add_library(gvtCore 
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
	find_package(Manta REQUIRED)
	set(GVT_RENDER_LIBS ${GVT_RENDER_LIBS} ${MANTA_TARGET_LINK_LIBRARIES}) 

    find_package(Boost REQUIRED COMPONENTS thread timer system)
    include_directories(${Boost_INCLUDE_DIRS})
    set(GVT_RENDER_LIBS ${GVT_RENDER_LIBS} ${Boost_LIBRARIES})
	
    if (GVT_RENDER_ADAPTER_OPTIX_PRIME)
    	find_package(CUDA REQUIRED)
    	find_package(Optix REQUIRED)

    	include_directories(${CUDA_INCLUDE_DIRS})
    	include_directories(${Optix_INCLUDE})
    	set(GVT_RENDER_LIBS ${GVT_RENDER_LIBS} ${CUDA_LIBRARIES} ${OptiX_LIBRARIES})
    	set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
    		--compiler-options -fno-strict-aliasing -lineinfo -use_fast_math
    		-gencode arch=compute_30,code=sm_30
    		-gencode arch=compute_35,code=sm_35
    		)
    	cuda_compile(GVT_RENDER_CUDA_OBJS ${GVT_RENDER_CUDA_HDRS} ${GVT_RENDER_CUDA_SRCS})
    	set(GVT_RENDER_SRCS ${GVT_RENDER_SRCS}
    		${GVT_RENDER_CUDA_OBJS}
    		)
    endif(GVT_RENDER_ADAPTER_OPTIX_PRIME)

	add_library(gvtRender 
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


if (GVT_RENDER_APP)
    find_package(MPI REQUIRED)
    include_directories(${MPI_INCLUDE_PATH})
    set(GVTAPP_RENDER_LIBS ${GVTAPP_RENDER_LIBS} ${MPI_LIBRARIES})

    find_package(Boost REQUIRED COMPONENTS thread timer system)
    include_directories(${Boost_INCLUDE_DIRS})
    set(GVTAPP_RENDER_LIBS ${GVTAPP_RENDER_LIBS} ${Boost_LIBRARIES})

	set(GVTAPP_RENDER_HDRS ${GVTAPP_RENDER_HDRS}
		src/apps/render/ConfigFileLoader.h
		src/apps/render/MantaRayTracer.h
		)

	set(GVTAPP_RENDER_SRCS ${GVTAPP_RENDER_SRCS}
		src/apps/render/ConfigFileLoader.cpp
		src/apps/render/GVTTrace.cpp
		src/apps/render/MantaRayTracer.cpp
		)

	add_executable(gvtTracer
		${GVTAPP_RENDER_HDRS}
		${GVTAPP_RENDER_SRCS}
		)

	target_link_libraries(gvtTracer ${GVTAPP_RENDER_LIBS} gvtCore gvtRender)

	install(TARGETS gvtTracer
		RUNTIME DESTINATION bin
		)
endif(GVT_RENDER_APP)
