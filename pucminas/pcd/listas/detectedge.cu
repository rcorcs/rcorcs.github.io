#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <iostream>
using namespace std;

#include <omp.h>

#include "img.h"
#include "imgio.h"


__global__ void greyScale(Image dst, Image src){
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	if(i<dst.height && j<dst.width){
		GetPixel(dst,i,j) = ((GetRGB(src,i,j,0)+GetRGB(src,i,j,1)+GetRGB(src,i,j,2))/3)&0xFF;
	}
}


__global__ void edgeDetection(Image dst, Image src){
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	if(i<dst.height && j<dst.width){
		int pX =
		GetPixel(src,i-1,j-1)*(-1)  +  GetPixel(src,i-1,j+1)*1 +
		GetPixel(src,i,j-1)  *(-2)  +  GetPixel(src,i,j+1)  *2 +
		GetPixel(src,i+1,j-1)*(-1)  +  GetPixel(src,i+1,j+1)*1 ;
		int pY = 
		GetPixel(src,i-1,j-1)*(1) + GetPixel(src,i-1,j)*(2) + GetPixel(src,i-1,j+1)*(1) +
		GetPixel(src,i+1,j-1)*(-1) + GetPixel(src,i+1,j)*(-2) + GetPixel(src,i+1,j+1)*(-1) ;

		int pixel = abs(int(pX))+abs(int(pY));
		GetPixel(dst,i,j) = 0xFF-pixel&0xFF;
	}
}

int main(int argc, char **argv){
	//Image img = loadPNMImage(argv[1]);
	//Image img = randomImage(4096, 2160 ,3); //4K resolution
	Image img = randomImage(7680, 4320 ,3); //8K resolution

	Image dimg;
	dimg.width = img.width;
	dimg.height = img.height;
	dimg.channel = img.channel;
	cudaMalloc(&(dimg.data),dimg.width*dimg.height*sizeof(int));

	Image grey;
	grey.width = img.width;
	grey.height = img.height;
	grey.channel = 1;
	cudaMalloc(&(grey.data),grey.width*grey.height*sizeof(int));
	
	Image edge;
	edge.width = grey.width;
	edge.height = grey.height;
	edge.channel = 1;
	cudaMalloc(&(edge.data),edge.width*edge.height*sizeof(int));

	double start_time = omp_get_wtime();
	//data transfer must be considered for the speedup
	cudaMemcpy(dimg.data,img.data,dimg.width*dimg.height*sizeof(int),cudaMemcpyHostToDevice);
	
	dim3 blockSize(16,16);
	dim3 nBlocks(dimg.height/blockSize.x + 1, dimg.width/blockSize.y + 1);
	//convert image to grey scale
	greyScale<<<nBlocks,blockSize>>>(grey,dimg);
	edgeDetection<<<nBlocks,blockSize>>>(edge,grey);
	
	//data transfer must be considered for the speedup
	cudaMemcpy(img.data,edge.data,img.width*img.height*sizeof(int),cudaMemcpyDeviceToHost);
	double time = omp_get_wtime() - start_time;
	cout << "Time: " << time << endl;

	img.channel = 1;
	//storePNMImage(img, argv[2]);
	free(img.data);
	cudaFree(dimg.data);
	cudaFree(grey.data);
	cudaFree(edge.data);
	return 0;
}
