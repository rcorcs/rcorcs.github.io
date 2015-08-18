
#include <stdio.h>

#include <omp.h>

int main(){
	printf("Hello\n");
	#pragma omp parallel num_threads(4)
	{
		//This command is executed by all threads
		printf("Entry #1 is printed by thread: %d\n", omp_get_thread_num());
        
		#pragma omp for
		for(int i = 0; i<3; i++){
			switch(i){
			case 0:
				printf("Inside section 1 with thread: %d\n", omp_get_thread_num());
				for(int i = 0; i<10; i++){
					printf("sec. 1: iteration %d by thread: %d\n", i, omp_get_thread_num());		
				}
				break;
			case 1:
				printf("Inside section 2 with thread: %d\n", omp_get_thread_num());
				for(int i = 0; i<10; i++){
					printf("sec. 2: iteration %d by thread: %d\n", i, omp_get_thread_num());		
				}
				break;
			case 2:
				printf("Inside section 3 with thread: %d\n", omp_get_thread_num());
				for(int i = 0; i<10; i++){
					printf("sec. 3: iteration %d by thread: %d\n", i, omp_get_thread_num());		
				}
				break;
			}
		}
		//This command is executed by all threads
		printf("Entry #4 is printed by thread: %d\n", omp_get_thread_num());
	}
	printf("Goodbye\n");
	return 0;
}
