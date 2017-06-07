#!/bin/bash

if [ -z ${GVT_MODELS} ]; then
  echo "GVT_MODELS is unset. Please specify the model directory."
  exit 
fi

GVT_MODELS_DIR=/work/03378/hpark/maverick/models

BIN_DIR="bin"
PLY_APPS="gvtPly gvtPlyNS"
OBJ_APPS="gvtFileLoad"

ADAPTERS="embree embree-stream"
SCHEDULERS="image domain"

IMAGE_SIZE="1024,1024"

# test ply apps on enzo black/white
for app in $PLY_APPS
do
  for adapter in $ADAPTERS
  do
    for scheduler in $SCHEDULERS
    do
      cmd="$BIN_DIR/$app -$adapter -$scheduler -wsize $IMAGE_SIZE \
      -file $GVT_MODELS/enzobw8/"
      $cmd
      echo $cmd
      mv -f output.ppm enzobw_${app}_${adapter}_${scheduler}.ppm
      echo "generated enzobw_${app}_${adapter}_${scheduler}.ppm"
    done
  done
done

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
      $cmd
      echo $cmd
      mv -f output.ppm enzocolor_${app}_${adapter}_${scheduler}.ppm
      echo "generated enzocolor_${app}_${adapter}_${scheduler}.ppm"
    done
  done
done

# test fileload app on bunny

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
      $cmd
      echo $cmd
      mv -f output.ppm cornell_${app}_${adapter}_${scheduler}.ppm
      echo "generated output.ppm cornell_${app}_${adapter}_${scheduler}.ppm"
    done
  done
done

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
      $cmd
      echo $cmd
      mv -f output.ppm ws2bonds_${app}_${adapter}_${scheduler}.ppm
      echo "generated output.ppm ws2bonds_${app}_${adapter}_${scheduler}.ppm"
    done
  done
done

