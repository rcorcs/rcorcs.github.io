/**
   Problema: cria um numero muito grande de tarefas (uma tarefa eh criada para cada chamada de pfib)
**/

#include<stdio.h>
#include<omp.h>

long fib(long n){
	long left, right;
	if(n<0) return 0;
	else if(n==1) return 1;
	else{
		#pragma omp task shared(left) firstprivate(n)
		{ left = fib(n-1); }
	
		#pragma omp task shared(right) firstprivate(n)
		{ right = fib(n-2); }
			
		#pragma omp taskwait
		return left+right;
	}
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
