#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <iostream>
using namespace std;

#include <omp.h>

#include "img.h"
#include "imgio.h"

int main(int argc, char **argv){
	Image img = loadPNMImage(argv[1]);
	//Image img = randomImage(4096, 2160 ,3); //4K resolution
	//Image img = randomImage(7680, 4320 ,3); //8K resolution
	Image grey;
	grey.width = img.width;
	grey.height = img.height;
	grey.channel = 1;
	grey.data = (int*)malloc(grey.width*grey.height*sizeof(int));

	Image edge;
	edge.width = grey.width;
	edge.height = grey.height;
	edge.channel = 1;
	edge.data = (int*)malloc(edge.width*edge.height*sizeof(int));

	double start_time = omp_get_wtime();

	//convert image to grey scale
	for(int i = 0; i<grey.height; i++){
		for(int j = 0; j<grey.width; j++){
			GetPixel(grey,i,j) = ((GetRGB(img,i,j,0)+GetRGB(img,i,j,1)+GetRGB(img,i,j,2))/3)&0xFF;
		}
	}

	//edge detection
	for(int i = 1; i<grey.height-1; i++){
		for(int j = 1; j<grey.width-1; j++){
			int pX =
				GetPixel(grey,i-1,j-1)*(-1)  +  GetPixel(grey,i-1,j+1)*1 +
				GetPixel(grey,i,j-1)  *(-2)  +  GetPixel(grey,i,j+1)  *2 +
				GetPixel(grey,i+1,j-1)*(-1)  +  GetPixel(grey,i+1,j+1)*1 ;
			int pY = 
				GetPixel(grey,i-1,j-1)*(1) + GetPixel(grey,i-1,j)*(2) + GetPixel(grey,i-1,j+1)*(1) +
				GetPixel(grey,i+1,j-1)*(-1) + GetPixel(grey,i+1,j)*(-2) + GetPixel(grey,i+1,j+1)*(-1) ;
			int pixel = abs(int(pX))+abs(int(pY));
			GetPixel(edge,i,j) = pixel&0xFF;
		}
	}

	//colour negative
	for(int i = 0; i<edge.height; i++){
		for(int j = 0; j<edge.width; j++){
			GetPixel(edge,i,j) = 0xFF-GetPixel(edge,i,j);
		}
	}

	double time = omp_get_wtime() - start_time;
	cout << "Time: " << time << endl;

	storePNMImage(edge, argv[2]);
	free(img.data);
	free(grey.data);
	free(edge.data);
	return 0;
}
