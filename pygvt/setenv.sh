export gvt_DIR=$WORK/Projects/GraviT/Install
export package_DIR=$WORK/Packages
export embree_DIR=$package_DIR/embree
export MPI_DIR=$I_MPI_ROOT/
export IceT_LIB_DIR=$package_DIR/icet/build/lib

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$embree_DIR:$MPI_DIR/lib64:$IceT_LIB_DIR:$gvt_DIR/lib


echo "======================================================"
echo "Add to your LD_LIBRARY_PATH"
echo $embree_DIR/build:$MPI_DIR/lib64:$IceT_LIB_DIR:$gvt_DIR/lib
echo "======================================================"
