#include<stdio.h>

#include<map>
using std::map;

long fib(long n){
	static map<long,long> cache;
	if(n<0) return 0;
	else if(n==1) return 1;
	else {
		map<long,long>::iterator it = cache.find(n);
		if(it!=cache.end()){
			return it->second;
		}else{
			long val = fib(n-1)+fib(n-2);
			cache[n] = val;
			return val;
		}
	}
}

int main(){
	long n = 50;
	printf("fib(%ld) = %ld\n", n, fib(n));
	return 0;
}
