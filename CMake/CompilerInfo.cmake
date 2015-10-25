
#
#  Copyright (c) 2008 - 2009 NVIDIA Corporation.  All rights reserved.
#
#  NVIDIA Corporation and its licensors retain all intellectual property and proprietary
#  rights in and to this software, related documentation and any modifications thereto.
#  Any use, reproduction, disclosure or distribution of this software and related
#  documentation without an express license agreement from NVIDIA Corporation is strictly
#  prohibited.
#
#  TO THE MAXIMUM EXTENT PERMITTED BY APPLICABLE LAW, THIS SOFTWARE IS PROVIDED *AS IS*
#  AND NVIDIA AND ITS SUPPLIERS DISCLAIM ALL WARRANTIES, EITHER EXPRESS OR IMPLIED,
#  INCLUDING, BUT NOT LIMITED TO, IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#  PARTICULAR PURPOSE.  IN NO EVENT SHALL NVIDIA OR ITS SUPPLIERS BE LIABLE FOR ANY
#  SPECIAL, INCIDENTAL, INDIRECT, OR CONSEQUENTIAL DAMAGES WHATSOEVER (INCLUDING, WITHOUT
#  LIMITATION, DAMAGES FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF
#  BUSINESS INFORMATION, OR ANY OTHER PECUNIARY LOSS) ARISING OUT OF THE USE OF OR
#  INABILITY TO USE THIS SOFTWARE, EVEN IF NVIDIA HAS BEEN ADVISED OF THE POSSIBILITY OF
#  SUCH DAMAGES
#

# Sets some variables depending on which compiler you are using
#
# USING_GCC  : gcc is being used for C compiler
# USING_GPP  : g++ is being used for C++ compiler
# USING_ICC  : icc is being used for C compiler
# USING_ICPC : icpc is being used for C++ compiler
# USING_WINDOWS_CL : Visual Studio's compiler
# USING_WINDOWS_ICL : Intel's Windows compiler

# Have to set this variable outside of the top level IF statement,
# since CMake will break if you use it in an IF statement.

#MESSAGE("CMAKE_C_COMPILER = ${CMAKE_C_COMPILER}")
#MESSAGE("CMAKE_CXX_COMPILER = ${CMAKE_CXX_COMPILER}")

SET(OPTIX_COMPILER_NAME_REGEXPR "icc.*$")

IF(NOT CMAKE_COMPILER_IS_GNUCC)
  # This regular expression also matches things like icc-9.1
  IF   (CMAKE_C_COMPILER MATCHES ${OPTIX_COMPILER_NAME_REGEXPR})
    SET(USING_ICC TRUE)
    SET(USING_KNOWN_C_COMPILER TRUE)
  ENDIF(CMAKE_C_COMPILER MATCHES ${OPTIX_COMPILER_NAME_REGEXPR})
ELSE(NOT CMAKE_COMPILER_IS_GNUCC)
  SET(USING_GCC TRUE)
  SET(USING_KNOWN_C_COMPILER TRUE)
ENDIF(NOT CMAKE_COMPILER_IS_GNUCC)

SET(OPTIX_COMPILER_NAME_REGEXPR "icpc.*$")
IF(NOT CMAKE_COMPILER_IS_GNUCXX)
  IF   (CMAKE_CXX_COMPILER MATCHES ${OPTIX_COMPILER_NAME_REGEXPR})
    SET(USING_ICPC TRUE)
    SET(USING_KNOWN_CXX_COMPILER TRUE)

    EXEC_PROGRAM(${CMAKE_CXX_COMPILER} 
      ARGS --version 
      OUTPUT_VARIABLE TEMP)

    STRING(REGEX MATCH "([0-9\\.]+)"
      INTEL_COMPILER_VERSION
      ${TEMP})

    MARK_AS_ADVANCED(INTEL_COMPILER_VERSION)
  ENDIF(CMAKE_CXX_COMPILER MATCHES ${OPTIX_COMPILER_NAME_REGEXPR})
ELSE(NOT CMAKE_COMPILER_IS_GNUCXX)
  SET(USING_GPP TRUE)
  SET(USING_KNOWN_CXX_COMPILER TRUE)
ENDIF(NOT CMAKE_COMPILER_IS_GNUCXX)

# The idea is to match a string that ends with cl but doesn't have icl in it.
SET(OPTIX_COMPILER_NAME_REGEXPR "([^i]|^)cl.*$")
IF(CMAKE_C_COMPILER MATCHES ${OPTIX_COMPILER_NAME_REGEXPR}
    AND CMAKE_CXX_COMPILER MATCHES ${OPTIX_COMPILER_NAME_REGEXPR})
  SET(USING_WINDOWS_CL TRUE)
  SET(USING_KNOWN_C_COMPILER TRUE)
  SET(USING_KNOWN_CXX_COMPILER TRUE)
  # We should set this macro as well to get our nice trig functions
  ADD_DEFINITIONS(-D_USE_MATH_DEFINES)
  # Microsoft does some stupid things like #define min and max.
  ADD_DEFINITIONS(-DNOMINMAX)
ENDIF(CMAKE_C_COMPILER MATCHES ${OPTIX_COMPILER_NAME_REGEXPR}
  AND CMAKE_CXX_COMPILER MATCHES ${OPTIX_COMPILER_NAME_REGEXPR})

# Intel compiler on windows.  Make sure this goes after the cl one.
SET(OPTIX_COMPILER_NAME_REGEXPR "icl.exe$")
IF(CMAKE_C_COMPILER MATCHES ${OPTIX_COMPILER_NAME_REGEXPR}
    AND CMAKE_CXX_COMPILER MATCHES ${OPTIX_COMPILER_NAME_REGEXPR})
  SET(USING_WINDOWS_ICL TRUE)
  SET(USING_WINDOWS_CL FALSE) # Turn off the other compiler just in case
  SET(USING_KNOWN_C_COMPILER TRUE)
  SET(USING_KNOWN_CXX_COMPILER TRUE)
  # We should set this macro as well to get our nice trig functions
  ADD_DEFINITIONS(-D_USE_MATH_DEFINES)
  # Microsoft does some stupid things like #define min and max.
  ADD_DEFINITIONS(-DNOMINMAX)
ENDIF(CMAKE_C_COMPILER MATCHES ${OPTIX_COMPILER_NAME_REGEXPR}
  AND CMAKE_CXX_COMPILER MATCHES ${OPTIX_COMPILER_NAME_REGEXPR})
# Do some error checking

# Mixing compilers
IF   (USING_ICC AND USING_GPP)
  FIRST_TIME_MESSAGE("Using icc combined with g++.  Good luck with that.")
ENDIF(USING_ICC AND USING_GPP)

IF   (USING_GCC AND USING_ICPC)
  FIRST_TIME_MESSAGE("Using gcc combined with icpc.  Good luck with that.")
ENDIF(USING_GCC AND USING_ICPC)

IF   (USING_ICC AND USING_GPP)
  FIRST_TIME_MESSAGE("Using icc combined with g++.  Good luck with that")
ENDIF(USING_ICC AND USING_GPP)

# Using unknown compilers
IF   (NOT USING_KNOWN_C_COMPILER)
  FIRST_TIME_MESSAGE("Specified C compiler ${CMAKE_C_COMPILER} is not recognized (gcc, icc).  Using CMake defaults.")
ENDIF(NOT USING_KNOWN_C_COMPILER)

IF   (NOT USING_KNOWN_CXX_COMPILER)
  FIRST_TIME_MESSAGE("Specified CXX compiler ${CMAKE_CXX_COMPILER} is not recognized (g++, icpc).  Using CMake defaults.")
ENDIF(NOT USING_KNOWN_CXX_COMPILER)

# Warn if the compiler is not icc on SGI_LINUX systems
IF   (CMAKE_SYSTEM_PROCESSOR MATCHES "ia64")
  IF(NOT USING_ICC)
	  FIRST_TIME_MESSAGE("Intel Compilers recommended on ia64. setenv CC icc before running cmake.")
  ENDIF(NOT USING_ICC)

  IF(NOT USING_ICPC)
	  FIRST_TIME_MESSAGE("Intel Compilers recommended on ia64. setenv CXX icpc before running cmake.")
  ENDIF(NOT USING_ICPC)
ENDIF(CMAKE_SYSTEM_PROCESSOR MATCHES "ia64")
