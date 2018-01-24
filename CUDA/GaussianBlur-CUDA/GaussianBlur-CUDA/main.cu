#include "cuda_runtime.h"
#include <cuda.h>
#include "device_launch_parameters.h"
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdio.h>
#include <memory.h>
#include <time.h>
#include "lodepng.h"

//kernel dimentions x and y
const int ker_x_dim = 5;
const int ker_y_dim = 5;
//sigma value for gaussian function
const double sigma = 1.0;
//declare kernel array (1d instead of 2 for efficiency)
float h_kernel[(ker_x_dim * 2)*(ker_y_dim * 2)];
__constant__ float d_kernel[(ker_x_dim * 2)*(ker_y_dim * 2)];

////
// Function to generate gaussian kernel values and store in h_kernel array
////
void getGaussianKernel() 
{
	double temp = 0.0;
	int r_i, r_j = 0;
	for (int i = -ker_x_dim; i <= ker_x_dim; i++) {
		r_i = i + ker_x_dim;
		for (int j = -ker_y_dim; j <= ker_y_dim; j++) {
			r_j = j + ker_y_dim;
			temp = exp(-((i*i) + (j*j)) / (2 * (sigma*sigma)));
			h_kernel[r_i*ker_y_dim+r_j] = temp / (2*M_PI*sigma*sigma);
		}
	}
	printf("Kernel generated successfully\n");
}
////
// Simple function to generate 1 dimentional index from an x and y value pair
// width, height: image width and height in pixels
// x, y: x and y values of the index 
////
__host__ __device__ int get1dIndex(int width, int height, int x, int y)
{
	if (x < 0) x = 0;
	if (x >= width) x = width - 1;
	if (y < 0) y = 0;
	if (y >= height) y = height - 1;
	return y*width + x;
}
////
// CUDA global function that runs the convolution on the image, by multiplying the kernel over each pixel
// input: pointer to the array that stores the image pixel RGB values on the device
// output: pointer to the empty array that will store the pixel RGB values in the device 
// width, height: image width and height in pixels
////
__global__ void runFilter(float* input, float* output, int width, int height) {
	// declare thread indecies 
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	float new_val = 0.0f;
	int r_i, r_j = 0;
	// check that thread aligns within the image dimentions
	if (col < width && row < height) {
		// run through the kernel matrix
		for (int i = -ker_x_dim; i < ker_x_dim; i++) {
			// get real kernel index 
			r_i = i + ker_x_dim;
			for (int j = -ker_y_dim; j < ker_y_dim; j++) {
				r_j = j + ker_y_dim;
				// get index image index
				int idx = get1dIndex(width, height, col + i, row + j);
				// work out new value by multiplying kernel value by pixel value
				new_val += d_kernel[r_i*ker_y_dim + r_j] * input[idx];
			}
		}
		// set new values to output array
		output[get1dIndex(width, height, col, row)] = new_val;
	}
}
////
// Function that allocates all device memory and runs the CUDA kernel
// input: pointer to the array that stores the image pixel RGB values on the host 
// output: pointer to the empty array that will store the pixel RGB values in the host  
// width, height: image width and height in pixels
////
void convolveImage(float* input, float* output, int width, int height) 
{
	float* d_input;
	float* d_output;
	// allocate memory in the devoce for the input and output pixel arrays
	cudaMalloc(&d_input, width*height * sizeof(float));
	cudaMalloc(&d_output, width*height * sizeof(float));
	// copy the values of the arrays stored on the host to the arrays stored on the device
	cudaMemcpyToSymbol(d_kernel, h_kernel, sizeof(h_kernel));
	cudaMemcpy(d_input, input, width*height * sizeof(float), cudaMemcpyHostToDevice);

	// declare block and grid dimentions
	dim3 blockDim(25, 25, 1);
	dim3 gridDim(width / (blockDim.x) + 1, height / (blockDim.y) + 1);
	printf("Image height: %d, width: %d\n", height, width);
	// run the CUDA kernel 
	runFilter << <gridDim, blockDim >> >(d_input, d_output, width, height);
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		// if there's an error, display it
		printf("Error: %s\n", cudaGetErrorString(cudaStatus));
	}
	cudaDeviceSynchronize();
	// copy the output from the device to the host, ready for png output
	cudaMemcpy(output, d_output, width*height * sizeof(float), cudaMemcpyDeviceToHost);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		// if there's an error, display it
		printf("Error running kernel: %s\n", cudaGetErrorString(cudaStatus));
	}
}

int main(int argc, int** argv) 
{
	// declare image paths 
	const char* image_path = "image.png";
	const char* output_path = "output.png";

	// create gaussian kernel 
	getGaussianKernel();

	std::vector<unsigned char> img_vect;
	unsigned int width, height;
	// import image into img_vect
	unsigned error = lodepng::decode(img_vect, width, height, image_path);
	if (error) {
		// if there's an error, display it
		printf("decoder error: %d, %s", error, lodepng_error_text(error));
	}
	int image_size = width*height; 
	float* temp;
	float* input;
	float* output;
	// allocate memory on the host for the image data
	cudaMallocHost(&temp, (image_size*3) * sizeof(float));
	cudaMallocHost(&input, (image_size) * sizeof(float));
	cudaMallocHost(&output, (image_size) * sizeof(float));
	int count = 0;
	// getting rid of the apha channel as it is not needed
	for (int i = 0; i < img_vect.size(); ++i) {
		if ((i + 1) % 4 != 0) {
			temp[count] = img_vect.at(i);
			count++;
		}
	}
	// generate grayscale by getting the mean of the RGB values and storing in one pixel value
	for (int i = 0; i < image_size; i++) {
		input[i] = (
			temp[i * 3 + 0] +
			temp[i * 3 + 1] +
			temp[i * 3 + 2])/3;
	}
	// start the timer 
	clock_t tStart = clock();
	// run the image convolution
	convolveImage(input, output, width, height);
	// end timer 
	clock_t tEnd = clock();
	// get time taken in milliseconds
	float ms = 1000.0f * (tEnd - tStart) / CLOCKS_PER_SEC;
	printf("Convolution took %fms.\n", ms);
	
	// image vector for lodepng output
	std::vector<unsigned char> out_image;
	for (int i = 0; i < image_size; i++) {
		out_image.push_back(output[i]);
		out_image.push_back(output[i]);
		out_image.push_back(output[i]);
		out_image.push_back(255);
	}
	// output image vector using lodepng
	error = lodepng::encode(output_path, out_image, width, height);
	if (error) {
		//if there's an error, display it
		printf("lodepng error: %s\n", lodepng_error_text(error));
	} else {
		printf("output image generated: %s\n", output_path);
	}
}

