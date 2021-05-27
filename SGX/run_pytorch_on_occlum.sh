#!/bin/bash
set -e

BLUE='\033[1;34m'
NC='\033[0m'

alpine_fs="/root/alpine_pytorch"

if [ ! -d $alpine_fs ];then
    echo "Error: cannot stat '$alpine_fs' directory"
    exit 1
fi


# 1. Init Occlum Workspace
rm -rf occlum_instance
[ -d occlum_instance ] || mkdir occlum_instance
cd occlum_instance
[ -d image ] || occlum init

# 2. Copy files into Occlum Workspace and build
if [ ! -d "image/lib/python3.7" ];then
    cp -f $alpine_fs/usr/bin/python3.7 image/bin
    cp -rf $alpine_fs/usr/lib/* image/lib
    cp -f $alpine_fs/lib/libz.so.1 image/lib
fi

