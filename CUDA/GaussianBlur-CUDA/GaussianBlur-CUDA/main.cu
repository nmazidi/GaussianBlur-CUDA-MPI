
#include <stdio.h>
#define _USE_MATH_DEFINES
#include <math.h>

const int ker_x_dim = 3;
const int ker_y_dim = 3;
const double sigma = 1.0;

void getGaussianKernel() {
	double *kernel = new double[ker_x_dim*ker_y_dim];

	//printf("%f, ", kernel[0][0]);
	// generate gaussian kernel values
	double temp, sum = 0.0;
	for (int i = 0; i <= ker_x_dim; i++) {
		for (int j = 0; j <= ker_y_dim; j++) {
			temp = exp(-((i*i) + (j*j)) / (2 * (sigma*sigma)));
			kernel[i*ker_y_dim+j] = temp / (2*M_PI*sigma*sigma);
			//sum += kernel[(i + ker_x_dim)*ker_y_dim + (j + ker_y_dim)];
			//printf("[%d][%d] = %f, ", i, j, kernel[i][j]);
		}
	}
	printf("%f", sum);
	/*for (int i = -ker_x_dim; i <= ker_x_dim; i++) {
		for (int j = -ker_y_dim; j <= ker_y_dim; j++) {
			kernel[(i + ker_x_dim)*ker_y_dim + (j + ker_y_dim)] /= sum;
			printf("[%d][%d] = %f, ", i, j, kernel[(i + ker_x_dim)*ker_y_dim + (j + ker_y_dim)]);
		}
	}*/
}

int main() {

	//const char* image_path = "image.png";
	//const char* output_path = "output.png";

	//declare 2d kernel array
	getGaussianKernel();

	//std::vector<unsigned char> img_vect;
	//unsigned int width, height;

	//unsigned error = lodepng::decode(img_vect, width, height, image_path);
	//if (error) std::cout << "decoder error " << error << ": " << lodepng_error_text(error) << std::endl;
	//unsigned char* input = new unsigned char[(img_vect.size() * 3) / 4];
	//unsigned char* output = new unsigned char[(img_vect.size() * 3) / 4];
	//int count = 0;
	//
	//// getting rid of the apha channel as it is not needed
	//for (int i = 0; i < img_vect.size(); ++i) {
	//	if ((i + 1) % 4 != 0) {
	//		input[count] = img_vect.at(i);
	//		output[count] = img_vect.at(i);
	//		count++;
	//	}
	//}
	//printf("%d, %d, %d\n", input[0], input[1], input[2]);
	//printf("%d, %d, %d\n", input[3], input[4], input[5]);
	//printf("%d, %d, %d\n", input[6], input[7], input[8]);
	//// Prepare data for output
	//std::vector<unsigned char> out_image;
	//printf("test");
	//for (int i = 0; i < img_vect.size(); ++i) {
	//	out_image.push_back(output[i]);
	//	if ((i + 1) % 3 == 0) {
	//		out_image.push_back(255);
	//	}
	//}
	//printf("test");

	//// Output the data
	////error = lodepng::encode(output_path, out_image, width, height);

	////if there's an error, display it
	////if (error) std::cout << "encoder error " << error << ": " << lodepng_error_text(error) << std::endl;

	
}

