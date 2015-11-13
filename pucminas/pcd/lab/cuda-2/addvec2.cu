
/**
PUC Minas
Prof. Rodrigo Caetano Rocha
**/

#include <stdio.h>
#include <cuda.h>

__global__ void addvec(float *dst, float *src1, float *src2, int n){
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if(i<n){
		dst[i] = src1[i] + src2[i];
	}
}

int main(){
	int n = 100;
	float *A = (float*)malloc(n*sizeof(float));
	float *B = (float*)malloc(n*sizeof(float));
	float *C = (float*)malloc(n*sizeof(float));
	float *dA, *dB, *dC;
	
	//memory space allocation on the GPU
	cudaMalloc(&dA, n*sizeof(float));
	cudaMalloc(&dB, n*sizeof(float));
	cudaMalloc(&dC, n*sizeof(float));

	for(int i = 0; i<n; i++){
		A[i] = i;
		B[i] = n-i;
	}

	//transfer the host vector to the GPU memory
	cudaMemcpy(dA, A, n*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dB, B, n*sizeof(float), cudaMemcpyHostToDevice);	
	
   int blockSize = 32;
   int nBlocks = n/blockSize + 1;
	addvec<<<nBlocks, blockSize>>>(dC,dA,dB,n);

	//transfer the vector back to the host memory
	cudaMemcpy(C, dC, n*sizeof(float), cudaMemcpyDeviceToHost);
	
	for(int i = 0; i<n; i++)
		printf("%.2f ", C[i]);
	printf("\b\n");

	free(A);
	free(B);
	free(C);
	//free the GPU memory space
	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dC);
}

