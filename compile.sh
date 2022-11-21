#! /bin/bash

if [ -e ./filter ] || [ -e ./kernels.o ];then
    echo "Pre-existent files, excluding"
    rm filter
    rm kernels.o
fi

nvcc -c kernels.cu
nvcc -ccbin g++ -Xcompiler "-std=c++11" kernels.o main.cpp lodepng.cpp helpers.cpp -lcuda -lcudart -o filter
