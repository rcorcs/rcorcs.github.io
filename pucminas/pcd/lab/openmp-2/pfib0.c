/**
   Problema: cria um numero muito grande de tarefas (uma tarefa eh criada para cada chamada de pfib)
**/

#include<stdio.h>
#include<omp.h>

int pfib(int n){
	int left, right;
	if(n<0) return 0;
	else if(n==1) return 1;
	else{
		#pragma omp task shared(left) firstprivate(n)
		{ left = pfib(n-1); }
	
		#pragma omp task shared(right) firstprivate(n)
		{ right = pfib(n-2); }
			
		#pragma omp taskwait
		return left+right;
	}
}

int main(){
	int n = 45;

	#pragma omp parallel shared(n) num_threads(12)
	{
		#pragma omp single
		{ printf("fib(%d) = %d\n", n, pfib(n)); }
	}

	return 0;
}
