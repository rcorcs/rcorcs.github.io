#include<stdio.h>
#include<omp.h>

int main(){
	int varShared = -1;
	int varPrivate = -1;
	
	#pragma omp parallel shared(varShared) private(varPrivate)
	{
		varShared = omp_get_thread_num();
		varPrivate = omp_get_thread_num();

		#pragma omp barrier

		printf("shared variable: %d\n",varShared);
		printf("private variable: %d\n",varPrivate);
	}

	printf("shared variable: %d\n",varShared);
	printf("private variable: %d\n",varPrivate);

	return 0;
}
