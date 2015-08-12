#include<stdio.h>

int main(){
    #pragma omp parallel num_threads(4)
    {
    	int i;
    	printf("Hello World!\n");
    	for(i=0;i<10;i++){
            printf("Iter: %d\n", i);
        }
    }
	printf("GoodBye World!\n");
	return 0;
}
