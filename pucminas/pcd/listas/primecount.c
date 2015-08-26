#include <stdio.h>
#include<math.h>

int is_prime(long num){
	if(num<=1) return 0;
	else if(num>3){
		if(num%2==0) return 0;
		long max_divisor = sqrt(num);
		for(long d = 3; d<=max_divisor; d+=2){
			if(num%d==0) return 0;
		}
	}
	return 1;
}

int main(){
	long max_num = 50000000L;
	long count_prime;
	long sum;

	if(max_num<=1) sum = 0;
	else if(max_num==2) sum = 1;
	else{
		sum = 1;//count the 2 as a prime, then start from 3
		for(int n = 3; n<max_num; n += 2){ //skip all even numbers
			count_prime = is_prime(n);
			sum = sum + count_prime;
		}
	}
	printf("maximum number checked: %ld\n", max_num);
	printf("number of primes: %ld\n", sum);

	return 0;
}
