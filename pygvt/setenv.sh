export gvt_DIR=$WORK/Projects/GraviT/Install2
export package_DIR=$WORK/Packages
export embree_DIR=$WORK/Projects/GraviT/third-party/galaxy/third-party/embree/install
export ospray_DIR=$WORK/Projects/GraviT/third-party/galaxy/third-party/ospray/install
export MPI_DIR=$I_MPI_ROOT/
#export IceT_LIB_DIR=$package_DIR/icet/build/lib
export IceT_LIB_DIR=$WORK/Projects/GraviT/third-party/icet/install/lib
export TBB_LIB=/opt/intel/compilers_and_libraries_2018.2.199/linux/tbb/lib/intel64/gcc4.7
#export TBB_LIB=/opt/intel/compilers_and_libraries_2017.4.196/linux/tbb/lib/intel64/gcc4.7
#export GregSpray_LIB_DIR=$gvt_DIR/../third-party/GregSpray
export Galaxy_LIB_DIR=$gvt_DIR/../third-party/galaxy/install/lib
export ospray_LIB_DIR=$ospray_DIR/lib64

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$embree_DIR/lib64:$MPI_DIR/lib64:$IceT_LIB_DIR:$gvt_DIR/lib:$ospray_DIR/lib64


echo "======================================================"
echo "Add to your LD_LIBRARY_PATH"
echo $embree_DIR/lib64:$MPI_DIR/lib64:$IceT_LIB_DIR:$gvt_DIR/lib
echo "======================================================"
