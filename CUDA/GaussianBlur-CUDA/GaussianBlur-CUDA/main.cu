#include "cuda_runtime.h"
#include <cuda.h>
#include "device_launch_parameters.h"
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdio.h>
#include <memory.h>
#include <time.h>
#include "lodepng.h"


const int ker_x_dim = 1;
const int ker_y_dim = 1;
const double sigma = 1.0;
//double *kernel = new double[(ker_x_dim * 2)*(ker_y_dim * 2)];
double h_kernel[(ker_x_dim * 2)*(ker_y_dim * 2)];
__constant__ double d_kernel[(ker_x_dim * 2)*(ker_y_dim * 2)];

void getGaussianKernel() 
{
	// generate gaussian kernel values
	double temp = 0.0;
	int r_i, r_j = 0;
	for (int i = -ker_x_dim; i <= ker_x_dim; i++) {
		r_i = i + ker_x_dim;
		for (int j = -ker_y_dim; j <= ker_y_dim; j++) {
			r_j = j + ker_y_dim;
			temp = exp(-((i*i) + (j*j)) / (2 * (sigma*sigma)));
			h_kernel[r_i*ker_y_dim+r_j] = temp / (2*M_PI*sigma*sigma);
			printf("[%d][%d] = %f, ", i, j, h_kernel[r_i*ker_y_dim+r_j]);
		}
	}
}

__host__ __device__ int get1dIndex(int width, int height, int x, int y)
{
	if (x < 0) x = 0;
	if (x >= width) x = width - 1;
	if (y < 0) y = 0;
	if (y >= height) y = height - 1;
	return y*width + x;
}

__global__ void runFilter(unsigned char* input, unsigned char* output, int width, int height) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	float r, g, b = 0;
	int count = 0;
	if (row < width && col < height) {
		for (int i = -ker_x_dim; i < ker_x_dim; i++) {
			for (int j = -ker_y_dim; j < ker_y_dim; j++) {
				int idx = get1dIndex(width, height, row + i, col + j);
				
			}
		}
	
		float r, g, b = 0.0f;
		for (int i = 0; i < ker_x_dim; i++) {
			
		}
	}

	printf("%f,%f,%f\n", r,g,b);

}

void convolveImage(float* input, float* output, int width, int height) 
{
	unsigned char* d_input;
	unsigned char* d_output;
	cudaMalloc(&d_input, width*height * sizeof(float));
	cudaMalloc(&d_output, width*height * sizeof(float));
	cudaMemcpyToSymbol(d_kernel, h_kernel, sizeof(h_kernel));
	cudaMemcpy(d_input, input, width*height * sizeof(double), cudaMemcpyHostToDevice);

	dim3 blockDim(25, 25, 1);
	dim3 gridDim((unsigned int) ceil((double)(width*height*3/blockDim.x)), 1, 1);
	printf("height: %d, width: %d", height, width);

	runFilter << <gridDim, blockDim >> >(d_input, d_output, width, height);
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		printf("Error: %s\n", cudaGetErrorString(cudaStatus));

	}
	cudaDeviceSynchronize();
}

int main(int argc, int** argv) 
{
	const char* image_path = "image.png";
	const char* output_path = "output.png";

	// create gaussian kernel 
	getGaussianKernel();

	std::vector<unsigned char> img_vect;
	unsigned int width, height;

	unsigned error = lodepng::decode(img_vect, width, height, image_path);
	if (error) {
		printf("decoder error: %d, %s", error, lodepng_error_text(error));
	}
	int image_size = width*height; 
	float* input;
	float* output;
	cudaMallocHost(&input, (image_size*3) * sizeof(float));
	cudaMallocHost(&output, (image_size*3) * sizeof(float));
	int count = 0;
	// getting rid of the apha channel as it is not needed
	for (int i = 0; i < img_vect.size(); ++i) {
		if ((i + 1) % 4 != 0) {
			input[count] = img_vect.at(i);
			count++;
		}
	}

	clock_t tStart = clock();
	convolveImage(input, output, width, height);
	clock_t tEnd = clock();
	float ms = 1000.0f * (tEnd - tStart) / CLOCKS_PER_SEC;
	printf("Convolution took %fms.\n", ms);

	std::vector<unsigned char> out_image;
	unsigned char temp;
	for (int i = 0; i < image_size * 3; i++) {
		out_image.push_back(input[i]);
		if ((i + 1) % 3 == 0) {
			out_image.push_back(255);
		}
	}
	error = lodepng::encode(output_path, out_image, width, height);

	//if there's an error, display it
	if (error) {
		printf("%s", lodepng_error_text(error));
	}
	
}

