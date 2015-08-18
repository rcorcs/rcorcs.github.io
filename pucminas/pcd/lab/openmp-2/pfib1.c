#include<stdio.h>
#include<omp.h>

/**
  Solucao: limita o numero de tarefas criadas 
**/
long pfib(const long n, const long tasks){
	long left, right;
	if(n<0) return 0;
	else if(n==1) return 1;
	else{
		if(tasks<=1){
			return pfib(n-1,1)+pfib(n-2,1);
		}else{
			#pragma omp task shared(left) firstprivate(n,tasks)
			{ left = pfib(n-1,tasks/2); }
	
			#pragma omp task shared(right) firstprivate(n,tasks)
			{ right = pfib(n-2,tasks/2); }
			
			#pragma omp taskwait
			return left+right;
		}
	}
}

long fib(long n){
	return pfib(n,omp_get_num_procs());
}

int main(){
	long n = 50;

	#pragma omp parallel shared(n)
	{
		#pragma omp single
		{ printf("fib(%ld) = %ld\n", n, fib(n)); }
	}

	return 0;
}
