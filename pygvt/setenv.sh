export embree_DIR=/opt/embree/2.16.1/
export gvt_DIR=/home/jbarbosa/local/gvt/
export MPI_DIR=$I_MPI_ROOT/
export IceT_LIB_DIR=/home/jbarbosa/d/icet/build/lib

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$embree_DIR/lib64:$MPI_DIR/lib64:$IceT_LIB_DIR:$gvt_DIR/lib


echo "======================================================"
echo "Add to your LD_LIBRARY_PATH"
echo $embree_DIR/lib64:$MPI_DIR/lib64:$IceT_LIB_DIR:$gvt_DIR/lib
echo "======================================================"
