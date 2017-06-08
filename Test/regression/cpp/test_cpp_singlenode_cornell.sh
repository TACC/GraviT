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

# test fileload app on cornell box
for app in $OBJ_APPS
do
  for adapter in $ADAPTERS
  do
    for scheduler in $SCHEDULERS
    do
      cmd="$BIN_DIR/$app -$adapter -$scheduler -wsize $IMAGE_SIZE \
      -obj $GVT_MODELS/cornell/cornell_box.obj \
      -eye 278.0,274.4,-800 -look 278.0,274.4,279.6 -point-light 10,300,-800,1000,1000,1000 -output output"
      echo $cmd
      $cmd
      mv -f output.ppm cornell_${app}_${adapter}_${scheduler}.ppm
      echo "generated output.ppm cornell_${app}_${adapter}_${scheduler}.ppm"
    done
  done
done
