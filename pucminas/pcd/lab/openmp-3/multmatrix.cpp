#include<time.h>
#include<stdio.h>
#include<stdlib.h>
#include<omp.h>

#include "array.h"

//   |a a|             |c c c|
//   |a a|   |b b b|   |c c c|
//   |a a| * |b b b| = |c c c|
//   |a a|             |c c c|
int main(){
	bool verbose = false;

	array2d<int> a(2, 4);
	array2d<int> b(3, 2);
	array2d<int> c(b.width(), a.height()); //c = a*b
	
	srand(time(NULL));

	//atribui valores aos vetores
	for(int h = 0; h<a.height(); h++)
		for(int w = 0; w<a.width(); w++)
			a(h,w) = rand()%10;

	for(int h = 0; h<b.height(); h++)
		for(int w = 0; w<b.width(); w++)
			b(h,w) = rand()%10;

	for(int h = 0; h<c.height(); h++)
		for(int w = 0; w<c.width(); w++)
			c(h,w) = 0;
	
	//exibe os vetores
	if(verbose){
		for(int h = 0; h<a.height(); h++){
			for(int w = 0; w<a.width(); w++)
				printf("%d\t", a(h,w));
			printf("\n");
		}

		printf("\n");
	
		for(int h = 0; h<b.height(); h++){
			for(int w = 0; w<b.width(); w++)
				printf("%d\t", b(h,w));
			printf("\n");
		}

		printf("\n");
	}
	//realiza a multiplicacao
	for(int h = 0; h<a.height(); h++)
		for(int w = 0; w<b.width(); w++)
			for(int k = 0; k<a.width(); k++)
				c(h,w) += a(h,k)*b(k,w);


	//exibe resultado
	if(verbose){
		for(int h = 0; h<c.height(); h++){
			for(int w = 0; w<c.width(); w++)
				printf("%d\t", c(h,w));
			printf("\n");
		}
	
		printf("\n");
	}
	return 0;
}
