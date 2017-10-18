#!/bin/bash

############################
# path to models
############################

if [ -z ${GVT_MODELS} ]; then
  echo "GVT_MODELS is unset. Please specify the model directory."
  exit 
fi

############################
# parameters
############################

GVT_MODELS_DIR=/work/03378/hpark/maverick/models

BIN_DIR="bin"
PLY_APPS="gvtPly gvtPlyNS"
OBJ_APPS="gvtFileLoad"

ADAPTERS="embree embree-stream"
SCHEDULERS="image domain"

IMAGE_SIZE="1024,1024"

############################
# test Ply
############################

# test ply apps on rm color
# WARNING: this is one of the most time consuming data
for app in $PLY_APPS
do
  for adapter in $ADAPTERS
  do
    for scheduler in $SCHEDULERS
    do
      cmd="$BIN_DIR/$app -$adapter -$scheduler -wsize $IMAGE_SIZE \
      -file $GVT_MODELS/rm8 \
      -eye 2660.41,-1657,-1548.52 -look 161.329,2133.88,2067.82 -point-light 2660.41,-1657,-1548.52,2500,2500,2500 "
      echo $cmd
      $cmd
      mv -f output.ppm rmcolor_${app}_${adapter}_${scheduler}.ppm
      echo "generated rmcolor_${app}_${adapter}_${scheduler}.ppm"
    done
  done
done
