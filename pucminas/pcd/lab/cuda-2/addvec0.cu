#include <stdio.h>

int main(){
	int n = 100;
	float *A = (float*)malloc(n*sizeof(float));
	float *B = (float*)malloc(n*sizeof(float));
	float *C = (float*)malloc(n*sizeof(float));
	
	for(int i = 0; i<n; i++){
		A[i] = i;
		B[i] = n-i;
	}

	for(int i = 0; i<n; i++)
		C[i] = A[i]+B[i];

	for(int i = 0; i<n; i++)
		printf("%.2f ", C[i]);
	printf("\b\n");

	free(A);
	free(B);
	free(C);
}

