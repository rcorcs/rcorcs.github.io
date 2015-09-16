#include <stdio.h>

#include <cuda.h>

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

        //transfer the vector back to the host memory
        cudaMemcpy(vec, dvec, n*sizeof(int), cudaMemcpyDeviceToHost);

        for(int i = 0; i<n; i++)
                printf("%d ", vec[i]);
        printf("\b\n");

	//zero the vector on the GPU memory space
	cudaMemset(dvec, 0, n*sizeof(n));

        //transfer the vector back to the host memory
        cudaMemcpy(vec, dvec, n*sizeof(int), cudaMemcpyDeviceToHost);

        for(int i = 0; i<n; i++)
                printf("%d ", vec[i]);
        printf("\b\n");

        free(vec);
        //free the GPU memory space
        cudaFree(dvec);
}

