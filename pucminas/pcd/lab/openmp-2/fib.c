
#include<stdio.h>

long fib(long n){
	if(n<0) return 0;
	else if(n==1) return 1;
	else return fib(n-1)+fib(n-2);
}

int main(){
	long n = 50;
	printf("fib(%ld) = %ld\n", n, fib(n));
	return 0;
}
