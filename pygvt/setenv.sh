export gvt_DIR=$WORK/Projects/GraviT/Install
export package_DIR=$WORK/Packages
export embree_DIR=$WORK/Projects/GraviT/third-party/embree/install
export MPI_DIR=$I_MPI_ROOT/
export IceT_LIB_DIR=$package_DIR/icet/build/lib
export TBB_LIB=/opt/intel/compilers_and_libraries_2017.4.196/linux/tbb/lib/intel64/gcc4.7
export GregSpray_LIB_DIR=$gvt_DIR/../third-party/GregSpray

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$embree_DIR/lib64:$MPI_DIR/lib64:$IceT_LIB_DIR:$gvt_DIR/lib


echo "======================================================"
echo "Add to your LD_LIBRARY_PATH"
echo $embree_DIR/build:$MPI_DIR/lib64:$IceT_LIB_DIR:$gvt_DIR/lib
echo "======================================================"
