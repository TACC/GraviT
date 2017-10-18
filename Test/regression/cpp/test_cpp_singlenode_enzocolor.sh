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

# test ply apps on enzo color
for app in $PLY_APPS
do
  for adapter in $ADAPTERS
  do
    for scheduler in $SCHEDULERS
    do
      cmd="$BIN_DIR/$app -$adapter -$scheduler -wsize $IMAGE_SIZE \
      -file $GVT_MODELS/enzocolor/ \
      -eye 1024,1024,1024 -look 0,0,0 -point-light 1024,1024,1024,500,500,500"
      echo $cmd
      $cmd
      mv -f output.ppm enzocolor_${app}_${adapter}_${scheduler}.ppm
      echo "generated enzocolor_${app}_${adapter}_${scheduler}.ppm"
    done
  done
done

