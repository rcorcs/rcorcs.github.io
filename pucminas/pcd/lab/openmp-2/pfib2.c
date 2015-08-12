
#include<stdio.h>
#include<omp.h>

int fib(const int n){
	if(n<0) return 0;
	else if(n==1) return 1;
	else return fib(n-1)+fib(n-2);
}

/**
  Solucao: limita o numero de tarefas criadas;
	   evita o teste condicional de tasks chamando fib(n)
**/
int pfib(const int n, const int tasks){
	int left, right;
	if(n<0) return 0;
	else if(n==1) return 1;
	else{
		if(tasks<=1){
			return fib(n-1)+fib(n-2);
		}else{
			#pragma omp task shared(left)
			{ left = pfib(n-1,tasks/2); }
	
			#pragma omp task shared(right)
			{ right = pfib(n-2,tasks/2); }
			
			#pragma omp taskwait
			return left+right;
		}
	}
}

int main(){
	int n = 45;

	#pragma omp parallel shared(n)
	{
		#pragma omp single
		{ printf("fib(%d) = %d\n", n, pfib(n,omp_get_num_threads())); }
	}

	return 0;
}
