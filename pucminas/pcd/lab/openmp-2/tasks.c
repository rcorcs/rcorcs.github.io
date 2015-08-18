
#include <stdio.h>

#include <omp.h>

int main(){
	printf("Hello\n");
	#pragma omp parallel num_threads(4) //four threads will be executed in total
	{
		//This command is executed by all threads
		//These commands here can also be executed concurrently with the following tasks
		printf("Entry #1 is printed by thread: %d\n", omp_get_thread_num());

		#pragma omp single
		{
			//This command is executed only once (by the thread 'single')
			printf("Entry #2 is printed by thread: %d\n", omp_get_thread_num());

			//The following tasks are executed concurrently (in parallel)
			#pragma omp task
			{
				//All these commands are executed only once by a selected thread
				printf("Inside task 1 with thread: %d\n", omp_get_thread_num());
				for(int i = 0; i<10; i++){
					printf("sec. 1: iteration %d by thread: %d\n", i, omp_get_thread_num());		
				}
			}
			//commands here ARE allowed and are executed only once by the thread 'single'
			#pragma omp task
			{
				//All these commands are executed only once by a selected thread
				printf("Inside task 2 with thread: %d\n", omp_get_thread_num());
				for(int i = 0; i<10; i++){
					printf("sec. 2: iteration %d by thread: %d\n", i, omp_get_thread_num());		
				}

			}
			//commands here ARE allowed and are executed only once by the thread 'single'
			#pragma omp task
			{
				//All these commands are executed only once by a selected thread
				printf("Inside task 3 with thread: %d\n", omp_get_thread_num());
				for(int i = 0; i<10; i++){
					printf("sec. 3: iteration %d by thread: %d\n", i, omp_get_thread_num());		
				}

			}
			
			//commands here ARE allowed
			//This command is executed only once (by the thread 'single')
			printf("Entry #3 is printed by thread: %d\n", omp_get_thread_num());

		}//There is a barrier at this point where all threads must be synchronised before going forward

		//This command is executed by all threads
		printf("Entry #4 is printed by thread: %d\n", omp_get_thread_num());
	}
	printf("Goodbye\n");
	return 0;
}
