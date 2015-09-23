#include <stdio.h>

int main(){
	int n = 10;
	
	//float *mat = new float[n*n];
	float *mat = (float*)malloc(n*n*sizeof(float));
	float *dmat;
	
	//memory space allocation on the GPU
	cudaMalloc(&dmat, n*n*sizeof(float));
	
	int count = 0;
	for(int i = 0; i<n; i++)
		for(int j = 0; j<n; j++)
			mat[i*n + j] = count++;

	//transfer the host matrix to the GPU memory
	cudaMemcpy(dmat, mat, n*n*sizeof(float), cudaMemcpyHostToDevice);

	//zero host matrix
	for(int i = 0; i<n; i++)
		for(int j = 0; j<n; j++)
			mat[i*n + j] = 0;


	//transfer the matrix back to the host memory
	cudaMemcpy(mat, dmat, n*n*sizeof(float), cudaMemcpyDeviceToHost);

	for(int i = 0; i<n; i++){
		for(int j = 0; j<n; j++)
			printf("%.2f ", mat[i*n + j]);
		printf("\b\n");
	}
	printf("\n");

	free(mat);
	cudaFree(dmat);
}

