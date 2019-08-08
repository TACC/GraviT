#!/bin/bash

SCRIPT_PATH="`dirname $0`"
# echo $SCRIPT_PATH

############################
# Ply
############################

$SCRIPT_PATH/test_cpp_singlenode_gl_enzobw.sh

$SCRIPT_PATH/test_cpp_singlenode_gl_enzocolor.sh

