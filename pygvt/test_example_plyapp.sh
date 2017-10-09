#!/bin/bash -x

if [ -z $GVT_DATA ]
then 
  GVT_DATA=$PWD/../data
fi

python example_plyapp.py --light-color 30 30 30 ${GVT_DATA}/wavelet_color

if [ -z ${GVT_MODELS} ]; then
   echo "GVT_MODELS is unset. Please specify the model directory."
else 
  python example_plyapp.py --light-color 900 900 900 ${GVT_MODELS}/enzocolor
fi

