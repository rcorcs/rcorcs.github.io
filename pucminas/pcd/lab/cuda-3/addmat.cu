#include <stdio.h>

#include <cuda.h>

__global__ void addvec(float *dst, float *src1, float *src2, int n){
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if(i<n){
		dst[i] = src1[i] + src2[i];
	}
}

int main(){
	int n = 10;
	float *A = (float*)malloc(n*n*sizeof(float));
	float *B = (float*)malloc(n*n*sizeof(float));
	float *C = (float*)malloc(n*n*sizeof(float));
	float *dA, *dB, *dC;
	
	//memory space allocation on the GPU
	cudaMalloc(&dA, n*n*sizeof(float));
	cudaMalloc(&dB, n*n*sizeof(float));
	cudaMalloc(&dC, n*n*sizeof(float));

	for(int i = 0; i<n; i++){
		for(int j = 0; j<n; j++){
			int idx = (i*n + j);
			A[idx] = idx;
			B[idx] = n*n-idx;
		}
	}

	//transfer the host matrix to the GPU memory
	cudaMemcpy(dA, A, n*n*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dB, B, n*n*sizeof(float), cudaMemcpyHostToDevice);	
	
	int blockSize = 32;
	int nBlocks = (n*n)/blockSize + 1;
	addvec<<<nBlocks,blockSize>>>(dC,dA,dB,n*n);

	//transfer the matrix back to the host memory
	cudaMemcpy(C, dC, n*n*sizeof(float), cudaMemcpyDeviceToHost);
	
	for(int i = 0; i<n; i++){
		for(int j = 0; j<n; j++)
			printf("%.2f ", C[i*n + j]);
		printf("\b\n");
	}
	printf("\n");

	free(A);
	free(B);
	free(C);
	//free the GPU memory space
	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dC);
}

