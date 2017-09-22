#!/bin/bash -x

if [ -z $GVT_DATA ]
then 
  GVT_DATA=$PWD/../data
fi

python example_reader.py --light-color 1 1 1 $GVT_DATA/geom/bunny.obj
python example_reader.py --light-color 30 30 30 $GVT_DATA/wavelet.vtp
python example_reader.py --light-color 500 500 500 $GVT_DATA/EnzoPlyData/block0.ply

