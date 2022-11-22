#include "kernels.h"
#include "helpers.h"
#include <iostream>
#include <cmath>


__global__
void blur(unsigned char* input_image, unsigned char* output_image, int width, int height) {

    const unsigned int offset = blockIdx.x*blockDim.x + threadIdx.x;

    int x = offset % width;
    int y = (offset-x)/width;

    int fsize = 5; // Filter size

    if(offset < width*height) {

        float output_red = 0;
        float output_green = 0;
        float output_blue = 0;

        int hits = 0;
        for(int ox = -fsize; ox < fsize+1; ++ox) {
            for(int oy = -fsize; oy < fsize+1; ++oy) {
                if((x+ox) > -1 && (x+ox) < width && (y+oy) > -1 && (y+oy) < height) {
                    const int currentoffset = (offset+ox+oy*width)*3;
                    output_red += input_image[currentoffset]; 
                    output_green += input_image[currentoffset+1];
                    output_blue += input_image[currentoffset+2];
                    hits++;
                }
            }
        }
        output_image[offset*3] = output_red/hits;
        output_image[offset*3+1] = output_green/hits;
        output_image[offset*3+2] = output_blue/hits;
    }
}

__global__
void detect_yellow(unsigned char* input_image, unsigned char* output_image, int width, int height) {

    const unsigned int offset = blockIdx.x + threadIdx.x * blockDim.x;

    int yellow_or_not_pixel;

    //verify if the pixel is yellow or around
    if(
    input_image[offset*3] >= 180 &&
    input_image[offset*3+1] >= 180 &&
    input_image[offset*3+2] <= 155
    ){
        yellow_or_not_pixel = 255;
    } else {
        yellow_or_not_pixel = 0;
    }

    output_image[offset*3] = yellow_or_not_pixel;
    output_image[offset*3+1] = yellow_or_not_pixel;
    output_image[offset*3+2] = yellow_or_not_pixel;

}


__global__
void negative(unsigned char* input_image, unsigned char* output_image, int width, int height) {

    const unsigned int offset = blockIdx.x + threadIdx.x * blockDim.x;
    
    output_image[offset*3] = 255 - input_image[offset*3];
    output_image[offset*3+1] = 255 - input_image[offset*3+1];
    output_image[offset*3+2] = 255 - input_image[offset*3+2];
    
}


__global__
void grayscale(unsigned char* input_image, unsigned char* output_image, int width, int height) {

    const unsigned int offset = blockIdx.x + threadIdx.x * blockDim.x;
    
    //origin: image processing classes && https://www.baeldung.com/cs/convert-rgb-to-grayscale

    int gray = (input_image[offset*3] * 0.3) + (input_image[offset*3+1] * 0.59) + (input_image[offset*3+2] * 0.11);
    output_image[offset*3] = gray;
    output_image[offset*3+1] = gray;
    output_image[offset*3+2] = gray;

}


__global__
void sepia(unsigned char* input_image, unsigned char* output_image, int width, int height) {

    const unsigned int offset = blockIdx.x + threadIdx.x * blockDim.x;
    
    //origin: https://www.geeksforgeeks.org/image-processing-in-java-colored-image-to-sepia-image-conversion/

    int newRed = (input_image[offset*3] * 0.393) + (input_image[offset*3+1] * 0.769) + (input_image[offset*3+2] * 0.189);
    int newGreen = (input_image[offset*3] * 0.349) + (input_image[offset*3+1] * 0.686) + (input_image[offset*3+2] * 0.168);
    int newBlue = (input_image[offset*3] * 0.272) + (input_image[offset*3+1] * 0.534) + (input_image[offset*3+2] * 0.131);
    
    newRed > 255 ? newRed = 255 : newRed = newRed;
    newGreen > 255 ? newGreen = 255 : newGreen = newGreen;
    newBlue > 255 ? newBlue = 255 : newBlue = newBlue;

    output_image[offset*3] = newRed;
    output_image[offset*3+1] = newGreen;
    output_image[offset*3+2] = newBlue;

}


__global__
void black_and_white(unsigned char* input_image, unsigned char* output_image, int width, int height) {

    const unsigned int offset = blockIdx.x + threadIdx.x * blockDim.x;

    int black_or_white_pixel;

    int average_rgb = (input_image[offset*3] + input_image[offset*3+1]  + input_image[offset*3+2] * 0.11) / 3;

    average_rgb >= 100 ? black_or_white_pixel = 255 : black_or_white_pixel = 0;

    output_image[offset*3] = black_or_white_pixel;
    output_image[offset*3+1] = black_or_white_pixel;
    output_image[offset*3+2] = black_or_white_pixel;

}


void filter (unsigned char* input_image, unsigned char* output_image, int width, int height, int filter_id) {

    unsigned char* dev_input;
    unsigned char* dev_output;
    getError(cudaMalloc( (void**) &dev_input, width*height*3*sizeof(unsigned char)));
    getError(cudaMemcpy( dev_input, input_image, width*height*3*sizeof(unsigned char), cudaMemcpyHostToDevice ));
 
    getError(cudaMalloc( (void**) &dev_output, width*height*3*sizeof(unsigned char)));

    dim3 blockDims(512,1,1);
    dim3 gridDims((unsigned int) ceil((double)(width*height*3/blockDims.x)), 1, 1 );

    switch (filter_id){
        case 0:
            blur<<<gridDims, blockDims>>>(dev_input, dev_output, width, height);
            break;
        case 1:
            black_and_white<<<gridDims, blockDims>>>(dev_input, dev_output, width, height);
            break;
        case 2:
            grayscale<<<gridDims, blockDims>>>(dev_input, dev_output, width, height);
            break;
        case 3:
            negative<<<gridDims, blockDims>>>(dev_input, dev_output, width, height);
            break;
        case 4:
            sepia<<<gridDims, blockDims>>>(dev_input, dev_output, width, height);
            break;
        case 5:
            detect_yellow<<<gridDims, blockDims>>>(dev_input, dev_output, width, height);
            break;
    }

    getError(cudaMemcpy(output_image, dev_output, width*height*3*sizeof(unsigned char), cudaMemcpyDeviceToHost ));

    getError(cudaFree(dev_input));
    getError(cudaFree(dev_output));

}

