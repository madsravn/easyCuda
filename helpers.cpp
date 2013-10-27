#include "helpers.h"
#include <iostream>
#include <cmath>

void getError(cudaError_t err) {
    if(err != cudaSuccess) {
        std::cout << "Error " << cudaGetErrorString(err) << std::endl;
    }
}

float* gaussianDistance(float sigma, const int fsize) {
    const int size = 2*fsize+1;
    float* kernel = new float[size*size]; 
    const float pi = std::atan(1.0f)*4.0f;
    float sigmasquared2 = 2*sigma*sigma;

    for(int x = -fsize; x < fsize+1; ++x) {
        for(int y = -fsize; y < fsize+1; ++y) {
            // (0,0) is center
            float f = expf(-(x*x/sigmasquared2 + y*y/sigmasquared2));
            kernel[x+fsize+(y+fsize)*size] = f / (sigmasquared2*pi);
        }
    }
    
    return kernel;
}


float* gaussianRange(float sigma, const int range) {
    float* kernel = new float[range];
    const float sqrt2pi = 2.0f*sqrt(std::atan(1.0f)*4.0f);
    float sigmasquared2 = 2*sigma*sigma;

    for(int x = 0; x < range; ++x) {
        float f = expf(-(x*x/sigmasquared2));
        kernel[x] = f / (sigma*sqrt2pi);
    }
    return kernel;
}

