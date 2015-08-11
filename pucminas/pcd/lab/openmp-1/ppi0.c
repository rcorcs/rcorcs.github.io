#include<stdio.h>

long long num_steps = 1000000000L;
double step;

int main(){
	int i;
	double x, pi, sum=0.0;
	step = 1.0/(double)num_steps;
	
    #pragma omp parallel for private(x)
	for(i=0; i<num_steps; i++){
		x = (i + 0.5)*step;
        
        #pragma omp critical
		sum = sum + 4.0/(1.0 + x*x);
	}

	pi = sum*step;
	
	printf("PI: %15.12f\n", pi);
	return 0;
}
