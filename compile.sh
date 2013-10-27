#! /bin/bash

rm filter
rm kernels.o
nvcc -c kernels.cu
nvcc -ccbin g++ -Xcompiler "-std=c++11" kernels.o main.cpp lodepng.cpp helpers.cpp -lcuda -lcudart -o filter
