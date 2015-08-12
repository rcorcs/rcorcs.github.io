#include<stdio.h>

long long num_steps = 1000000000L;
double step;

int main(){
	int i;
	double x, pi, sum, gsum=0.0;
	step = 1.0/(double)num_steps;
	
	#pragma omp parallel private(x,sum) shared(gsum)
	{
        //local variable for calculating the sum in each thread
        sum = 0.0;
        
    	#pragma omp for
    	for(i=0; i<num_steps; i++){
    		x = (i + 0.5)*step;
    		sum = sum + 4.0/(1.0 + x*x);
        }

    	#pragma omp critical
        { gsum = gsum + sum; } //adds the local sum to the global sum
	}

	pi = gsum*step; //use the global sum to calculate pi
	
	printf("PI: %15.12f\n", pi);
	return 0;
}
