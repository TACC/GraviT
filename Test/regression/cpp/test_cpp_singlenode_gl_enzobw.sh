#!/bin/bash

if [ -z ${GVT_MODELS} ]; then
  echo "GVT_MODELS is unset. Please specify the model directory."
  exit 
fi

GVT_MODELS_DIR=/work/03378/hpark/maverick/models

BIN_DIR="bin"
PLY_APPS="glTracer"

ADAPTERS="embree embree-stream"
SCHEDULERS="image domain"

IMAGE_SIZE="800,800"

# test ply apps on enzo black/white
for app in $PLY_APPS
do
  for adapter in $ADAPTERS
  do
    for scheduler in $SCHEDULERS
    do
      cmd="$BIN_DIR/$app -$adapter -$scheduler -wsize $IMAGE_SIZE \
      -file $GVT_MODELS/enzobw8/"
      echo $cmd
      $cmd
    done
  done
done

