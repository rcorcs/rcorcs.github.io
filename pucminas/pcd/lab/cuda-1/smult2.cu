#include <stdio.h>

#include <cuda.h>

__device__ int mult(int val, int factor){
        return val*factor;
}

__global__ void smult(int *dvec, int n, int factor){
        int i = threadIdx.x;
        if(i<n){
                dvec[i] = mult(dvec[i], factor);
        }
}

int main(){
        int n = 100;
        int *vec = (int*)malloc(n*sizeof(int));
        int *dvec;
        //memory space allocation on the GPU
        cudaMalloc(&dvec, n*sizeof(int));
        for(int i = 0; i<n; i++)
                vec[i] = i;

        //transfer the host vector to the GPU memory
        cudaMemcpy(dvec, vec, n*sizeof(int), cudaMemcpyHostToDevice);
        for(int i = 0; i<n; i++)
                vec[i] = 0;

	//multiply all elements by a factor of 2 on the GPU
	//scalar multiplication
        smult<<<1, n>>>(dvec, n, 2);
        //transfer the vector back to the host memory
        cudaMemcpy(vec, dvec, n*sizeof(int), cudaMemcpyDeviceToHost);

        for(int i = 0; i<n; i++)
                printf("%d ", vec[i]);
        printf("\b\n");

        free(vec);
        //free the GPU memory space
        cudaFree(dvec);
}

