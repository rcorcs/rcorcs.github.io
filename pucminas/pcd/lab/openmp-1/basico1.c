#include<stdio.h>
#include<omp.h>

int main(){
	printf("Number of threads before parallel: %d\n", omp_get_num_threads());
	#pragma omp parallel
	{
		printf("Number of threads after parallel: %d\n", omp_get_num_threads());

		printf("Hello thread %d!\n", omp_get_thread_num());

		if( omp_get_thread_num() == 1 ){
			printf("Different work done by thread %d\n", omp_get_thread_num());
		}

		#pragma omp single
		printf("Only one thread will do (thread %d).\n", omp_get_thread_num());

		#pragma omp for
		for(int i = 0; i<12; i++){
			printf("It. %d, thread %d\n", i, omp_get_thread_num());
		}
	}
	return 0;
}
