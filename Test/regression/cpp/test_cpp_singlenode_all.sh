#!/bin/bash

SCRIPT_PATH="`dirname $0`"
# echo $SCRIPT_PATH

############################
# Obj
############################

$SCRIPT_PATH/test_cpp_singlenode_cornell.sh

$SCRIPT_PATH/test_cpp_singlenode_ws2bonds.sh

############################
# Ply
############################

$SCRIPT_PATH/test_cpp_singlenode_enzobw.sh

$SCRIPT_PATH/test_cpp_singlenode_enzocolor.sh

$SCRIPT_PATH/test_cpp_singlenode_rm.sh

