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
# test Obj
############################

# test fileload app on ws2_bonds
for app in $OBJ_APPS
do
  for adapter in $ADAPTERS
  do
    for scheduler in $SCHEDULERS
    do
      cmd="$BIN_DIR/$app -$adapter -$scheduler -wsize $IMAGE_SIZE \
      -obj $GVT_MODELS/ws2_bonds/WS2_bonds.obj \
      -eye 0.1,0.4,0.001 -look -0.0020855,0.350707,0.002178 \
      -point-light 0.1,0.4,0.001,1,1,1 -output output"
      echo $cmd
      $cmd
      mv -f output.ppm ws2bonds_${app}_${adapter}_${scheduler}.ppm
      echo "generated output.ppm ws2bonds_${app}_${adapter}_${scheduler}.ppm"
    done
  done
done

