#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdio.h>
#include "lodepng.h"

const int ker_x_dim = 3;
const int ker_y_dim = 3;
const double sigma = 1.0;

double *kernel = new double[(ker_x_dim * 2)*(ker_y_dim * 2)];

void getGaussianKernel() {
	// generate gaussian kernel values
	double temp, sum = 0.0;
	int r_i, r_j = 0;
	for (int i = -ker_x_dim; i <= ker_x_dim; i++) {
		r_i = i + ker_x_dim;
		for (int j = -ker_y_dim; j <= ker_y_dim; j++) {
			r_j = j + ker_y_dim;
			temp = exp(-((i*i) + (j*j)) / (2 * (sigma*sigma)));
			kernel[r_i*ker_y_dim+r_j] = temp / (2*M_PI*sigma*sigma);
			printf("[%d][%d] = %f, ", i, j, kernel[r_i*ker_y_dim+r_j]);
		}
	}
}

__global__ void runFilter(float *input, float *output) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = -ker_x_dim; i < ker_x_dim; ++i) {
		for (int j = -ker_y_dim; j < ker_y_dim; ++j) {

		}
	}
}

int main() {
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
	unsigned char* input = new unsigned char[(img_vect.size() * 3) / 4];
	unsigned char* output = new unsigned char[(img_vect.size() * 3) / 4];
	int count = 0;
	
	// getting rid of the apha channel as it is not needed
	for (int i = 0; i < img_vect.size(); ++i) {
		if ((i + 1) % 4 != 0) {
			input[count] = img_vect.at(i);
			output[count] = img_vect.at(i);
			count++;
		}
	}
	//printf("%d, %d, %d\n", input[0], input[1], input[2]);

	dim3 blockDim(16, 16, 1);
	dim3 gridDim(width / blockDim.x + 1, height / blockDim.y + 1);

	runFilter << <gridDim, blockDim >> > (input, output);
	

	// Prepare data for output
	/*std::vector<unsigned char> out_image;
	printf("test");
	for (int i = 0; i < img_vect.size(); ++i) {
		out_image.push_back(output[i]);
		if ((i + 1) % 3 == 0) {
			out_image.push_back(255);
		}
	}*/

	// Output the data
	//error = lodepng::encode(output_path, out_image, width, height);

	//if there's an error, display it
	//if (error) std::cout << "encoder error " << error << ": " << lodepng_error_text(error) << std::endl;

	
}

