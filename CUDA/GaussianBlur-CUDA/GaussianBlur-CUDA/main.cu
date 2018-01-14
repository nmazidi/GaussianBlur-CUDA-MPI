#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
#include "lodepng.h"
#define _USE_MATH_DEFINES
#include <math.h>

const int ker_x_dim = 5;
const int ker_y_dim = 5;

void getGaussianKernel(double (&kernel)[ker_x_dim][ker_y_dim], double sigma) {
	//printf("%f, ", kernel[0][0]);
	// generate gaussian kernel values
	for (int i = 0; i < ker_x_dim; i++) {
		for (int j = 0; j < ker_y_dim; j++) {
			double temp = exp(-((i*i) + (j*j) / (2 * (sigma*sigma))));
			kernel[i][j] = temp / (2 * M_PI*(sigma*sigma));
			printf("[%d][%d] = %f, ", i, j, kernel[i][j]);
		}
	}
}

int main(int argc, char** argv) {

	const char* image_path = "image.png";
	const char* output_path = "output.png";


	//declare 2d kernel array
	double gaussian_kernel[ker_x_dim][ker_y_dim];
	getGaussianKernel(gaussian_kernel, 2);

	std::vector<unsigned char> img_vect;
	unsigned int width, height;

	unsigned error = lodepng::decode(img_vect, width, height, image_path);
	if (error) std::cout << "decoder error " << error << ": " << lodepng_error_text(error) << std::endl;
	unsigned char* input = new unsigned char[(img_vect.size() * 3) / 4];
	unsigned char* output = new unsigned char[(img_vect.size() * 3) / 4];
	int count = 0;
	printf("%d, %d, %d, %d\n", img_vect.at(0), img_vect.at(1), img_vect.at(2), img_vect.at(3));
	printf("%d, %d, %d, %d\n", img_vect.at(4), img_vect.at(5), img_vect.at(6), img_vect.at(7));
	printf("%d, %d, %d, %d\n", img_vect.at(8), img_vect.at(9), img_vect.at(10), img_vect.at(11));
	// getting rid of the apha channel as it is not needed
	for (int i = 0; i < img_vect.size(); ++i) {
		if ((i + 1) % 4 != 0) {
			input[count] = img_vect.at(i);
			output[count] = img_vect.at(i);
			count++;
		}
	}
	printf("%d, %d, %d\n", input[0], input[1], input[2]);
	printf("%d, %d, %d\n", input[3], input[4], input[5]);
	printf("%d, %d, %d\n", input[6], input[7], input[8]);
	// Prepare data for output
	std::vector<unsigned char> out_image;
	printf("test");
	for (int i = 0; i < img_vect.size(); ++i) {
		out_image.push_back(output[i]);
		if ((i + 1) % 3 == 0) {
			out_image.push_back(255);
		}
	}
	printf("test");

	// Output the data
	//error = lodepng::encode(output_path, out_image, width, height);

	//if there's an error, display it
	//if (error) std::cout << "encoder error " << error << ": " << lodepng_error_text(error) << std::endl;

	delete[] image_path;
	delete[] output_path;
	return 0;

}

