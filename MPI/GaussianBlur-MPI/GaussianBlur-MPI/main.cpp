#define _USE_MATH_DEFINES
#include <math.h>
#include <stdio.h>
#include <memory.h>
#include <time.h>
#include <mpi.h>
#include "lodepng.h"

//kernel dimentions x and y
const int ker_x_dim = 5;
const int ker_y_dim = 5;
//sigma value for gaussian function
const double sigma = 1.0;
//declare kernel array (1d instead of 2 for efficiency)
float kernel[(ker_x_dim * 2)*(ker_y_dim * 2)];

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
			kernel[r_i*ker_y_dim + r_j] = temp / (2 * M_PI*sigma*sigma);
		}
	}
	printf("Kernel generated successfully\n");
}
////
// Simple function to generate 1 dimentional index from an x and y value pair
// width, height: image width and height in pixels
// x, y: x and y values of the index 
////
int get1dIndex(int width, int height, int x, int y)
{
	if (x < 0) x = 0;
	if (x >= width) x = width - 1;
	if (y < 0) y = 0;
	if (y >= height) y = height - 1;
	return y*width + x;
}
int main(int argc, char **argv) {
	// declare image paths 
	const char* image_path = "image.png";
	const char* output_path = "output.png";
	float* temp;
	float* input;
	float* output;
	float** split_input;
	int image_size = 0;
	float buf;
	std::vector<unsigned char> img_vect;
	unsigned int width, height;
	int rank, size, next, prev;
	MPI_Status status;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if (size < 2) {
		if (rank == 0) printf("ERROR: size = %d", size);
		MPI_Finalize();
		exit(-1);
	}
	next = rank + 1;
	prev = rank - 1;

	if (next >= size) {
		next = MPI_PROC_NULL;
	}
	if (prev < 0) {
		prev = MPI_PROC_NULL;
	}
	if (rank == 0) {
		// import image into img_vect
		unsigned error = lodepng::decode(img_vect, width, height, image_path);
		if (error) {
			// if there's an error, display it
			printf("decoder error: %d, %s", error, lodepng_error_text(error));
		}
		image_size = width*height;
		getGaussianKernel();
	}
	MPI_Bcast(&image_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
	input = (float*)malloc((image_size) * sizeof(float));
	if (rank == 0) {
		// allocate memory on the host for the image data
		temp = (float*)malloc((image_size * 3) * sizeof(float));
		output = (float*)malloc((image_size) * sizeof(float));
		
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
				temp[i * 3 + 2]) / 3;
		}
		printf("Processing %d x %d image\n", width, height);
	}
	float* portion = (float*)malloc((image_size / size) * sizeof(float));
	MPI_Scatter(input, image_size / size, MPI_FLOAT, portion, image_size / size, MPI_FLOAT, 0, MPI_COMM_WORLD);
	printf("process %d: %f\n", rank, portion[65535]);



	MPI_Finalize();
	
}
