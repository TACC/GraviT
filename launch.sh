
#cmake -DBOOST_INCLUDEDIR=/work/01502/rri/opt/boost/include -DMANTA_SOURCE_DIR=/work/01502/rri/opt/Manta -DMANTA_BUILD_PREFIX=/work/01502/rri/opt/Manta/build -DOptiX_INSTALL_DIR=/work/01502/rri/opt/NVIDIA-OptiX-SDK-3.8.0-linux64 -DOptiX_INCLUDE=/work/01502/rri/opt/NVIDIA-OptiX-SDK-3.8.0-linux64/include -DEMBREE_BUILD_PREFIX=/work/01502/rri/opt/embree ..




cmake -DMPI_CXX_LIBRARIES=/opt/local/lib/openmpi-mp/libmpi.dylib -DMPI_CXX_INCLUDE_PATH=/opt/local/include/openmpi-mp -DMPI_C_LIBRARIES=/opt/local/lib/openmpi-mp/libmpi.dylib -DMPI_C_INCLUDE_PATH=/opt/local/include/openmpi-mp -DMANTA_SOURCE_DIR=/Users/rri/work/TACC-GraviT/Manta -DMANTA_BUILD_PREFIX=/Users/rri/work/TACC-GraviT/Manta/build -DOptiX_INSTALL_DIR=/Developer/OptiX -DOptiX_INCLUDE=/Developer/OptiX/include -DCUDA_PROPAGATE_HOST_FLAGS=OFF -dCUDA_SEPARABLE_COMPILATION=ON ..
