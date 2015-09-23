#include <stdio.h>

int main(){
	int n = 10;
	
	float **mat = new float*[n];
	for(int i = 0; i<n; i++){
		mat[i] = new float[n];
	}
	
	int count = 0;
	for(int i = 0; i<n; i++)
		for(int j = 0; j<n; j++)
			mat[i][j] = count++;

	for(int i = 0; i<n; i++){
		for(int j = 0; j<n; j++)
			printf("%.2f ", mat[i][j]);
		printf("\b\n");
	}
	printf("\n");

	for(int i = 0; i<n; i++){
		free(mat[i]);
	}
	free(mat);
}

