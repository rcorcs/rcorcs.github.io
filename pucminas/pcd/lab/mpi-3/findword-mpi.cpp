
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mpi.h>

#define MAX_FILE_NAME 1024
#define MAX_WORD 1024

int main(int argc, char **argv){
	int myrank, nprocs;
	char *searchWord = argv[1];

	char fileName[MAX_FILE_NAME];
	char word[MAX_WORD];

	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
	MPI_Comm_size(MPI_COMM_WORLD,&nprocs);

	printf("Process %d searching: %s\n", myrank, searchWord);

	sprintf(fileName, "input%d.txt", myrank);

	FILE *fd = fopen(fileName,"r");

	int count = 0;
	while(fscanf(fd,"%s",word)==1){
		if(!strcmp(word,searchWord)){
			count++;
		}
	}
	fclose(fd);		
	
	printf("Process %d found: %d\n", myrank, count);

	int globalcount = 0;
	MPI_Reduce(&count, &globalcount, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        if(myrank==0) printf("Total found: %d\n", globalcount);

	MPI_Finalize();
	return 0;
}
