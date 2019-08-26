#!/bin/bash
#
# This script runs cmake to configure the GraviT2 project
# run from within the build directory
#
PROJECTDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# now set the Install directory to...Install. which is in the PROJECTDIR
PROJECT="${PROJECTDIR}/Install"
#COMP=intel19
COMP=intel18
#COMP=gcc
if test "$COMP" == 'intel18' ; then
   module load intel/18.0.2
   CC=icc
   CXX=icpc
elif test "$COMP" == 'intel19' ; then
   module load intel/19.0.2
   CC=icc
   CXX=icpc
elif test "$COMP" == 'gcc' ; then
   module load gcc/7.1.0
   CC=gcc
   CXX=g++
fi
export CC CXX
cmake .. \
    -DCMAKE_INSTALL_PREFIX=$PROJECTDIR/Install 
cmake ..

