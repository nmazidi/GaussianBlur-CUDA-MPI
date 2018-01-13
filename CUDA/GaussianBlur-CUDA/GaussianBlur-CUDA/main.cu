#include <iostream>
#include <stdio.h>
#include "lodepng.h"


int main(int argc, char** argv) {

	const char* image_path = "image.png";
	//const char* output_path = argv[2];

	std::vector<unsigned char> in_image;
	unsigned int width, height;

	unsigned error = lodepng::decode(in_image, width, height, image_path);
	if (error) std::cout << "decoder error " << error << ": " << lodepng_error_text(error) << std::endl;
	unsigned char test;
	test = in_image.at(2);
	printf("test%d", test);
}