###############################################################################
# This scripts prompts the user to set the variable EMBREE_SOURCE_DIR and
# EMBREE_BUILD_PREFIX. After this is done, the script sets variables:
#
# EMBREE_INCLUDE               -- Paths containing Embree header files.
# EMBREE_TARGET_LINK_LIBRARIES -- List of Embree shared libraries.  (-l on link line)
# EMBREE_LINK_DIRECTORIES      -- Path containing shared libraries (-L on link line)
#
# Additionally several .cmake scripts from the Embree build are executed to
# insure a similar build environment will be used by the project.
#
###############################################################################
#SET(EMBREE_SOURCE_DIR ${CMAKE_SOURCE_DIR})
#SET(EMBREE_BUILD_PREFIX ${CMAKE_SOURCE_DIR}/buildTigger)
IF(NOT EMBREE_BUILD_PREFIX)
	SET(EMBREE_BUILD_PREFIX $ENV{EMBREE_BUILD_PREFIX})
ENDIF()

IF   (EMBREE_BUILD_PREFIX)

  # Set the include and link variables.
  SET(EMBREE_INCLUDE
    ${EMBREE_BUILD_PREFIX}/include
    )

  SET(EMBREE_TARGET_LINK_LIBRARIES
    embree
    tbb
    tbbmalloc
    )
IF(EXISTS  ${EMBREE_BUILD_PREFIX}/lib)
  SET(EMBREE_LINK_DIRECTORIES ${EMBREE_BUILD_PREFIX}/lib)
ENDIF()
IF(EXISTS  ${EMBREE_BUILD_PREFIX}/lib64)
  SET(EMBREE_LINK_DIRECTORIES ${EMBREE_BUILD_PREFIX}/lib64)
ENDIF()


  # Include Embree header files.
  INCLUDE_DIRECTORIES(
    ${EMBREE_INCLUDE}
    )

  # Include Embree library directory.
  LINK_DIRECTORIES(
    ${EMBREE_LINK_DIRECTORIES}
    )


# Otherwise prompt the user to enter the desired Embree build to use.
ELSE (EMBREE_BUILD_PREFIX)

  SET(EMBREE_BUILD_PREFIX "" CACHE PATH "Build directory containing lib/ bin/ etc. sub-directories.")

  MESSAGE(FATAL_ERROR "Manually set the paths EMBREE_SOURCE_DIR and EMBREE_BUILD_PREFIX")

ENDIF(EMBREE_BUILD_PREFIX)
