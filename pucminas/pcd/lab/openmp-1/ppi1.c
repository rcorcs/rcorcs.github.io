#include<stdio.h>

long long num_steps = 1000000000L;
double step;

int main(){
	int i;
	double x, pi, gsum=0.0, sum=0.0;
	step = 1.0/(double)num_steps;
	
	#pragma omp parallel private(x,sum) shared(gsum)
	{
    	#pragma omp for
    	for(i=0; i<num_steps; i++){
    		x = (i + 0.5)*step;
    		sum = sum + 4.0/(1.0 + x*x);
        }

    	#pragma omp critical
        { gsum = gsum + sum; }
	}

	pi = gsum*step;
	
	printf("PI: %15.12f\n", pi);
	return 0;
}
