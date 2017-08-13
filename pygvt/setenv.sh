export embree_DIR=/opt/embree/2.16.1/
export gvt_DIR=/home/jbarbosa/local/gvt/
export MPI_DIR=$I_MPI_ROOT/
export Boost_DIR=/opt/boost/1.63/
export IceT_LIB_DIR=/home/jbarbosa/d/icet/build/lib

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$embree_DIR/lib64:$MPI_DIR/lib64:$IceT_LIB_DIR:$Boost_DIR/lib:$gvt_DIR/lib


echo "======================================================"
echo "Add to your LD_LIBRARY_PATH"
echo $embree_DIR/lib64:$MPI_DIR/lib64:$IceT_LIB_DIR:$Boost_DIR/lib:$gvt_DIR/lib
echo "======================================================"
