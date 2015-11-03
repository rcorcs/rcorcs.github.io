#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

/*
! This program shows how to use MPI_Scatter and MPI_Gather
! Each processor gets different data from the root processor
! by way of mpi_scatter.  The data is summed and then sent back
! to the root processor using MPI_Gather.  The root processor
! then prints the global sum. 
*/

#define ROOTRANK 0

int main(int argc,char *argv[]){
	int *A, *B, *C;
	int *myA, *myB, *myC;
	int myrank, nprocs;

	int size, count;
	
	MPI_Init(&argc,&argv);
   MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

	/* each processor will get count elements from the root */
	count=4;

	myA=(int*)malloc(count*sizeof(int));
	myB=(int*)malloc(count*sizeof(int));
	myC=(int*)malloc(count*sizeof(int));

	/* create the data to be sent on the root */
	if(myrank == ROOTRANK){
	   size=count*nprocs;
		A=(int*)malloc(size*sizeof(int));
		B=(int*)malloc(size*sizeof(int));
		C=(int*)malloc(size*sizeof(int));
		for(int i=0;i<size;i++)
			A[i] = B[i] = i;
	}

	/* send different data to each processor */
	MPI_Scatter(A, count, MPI_INT, myA, count, MPI_INT,
	                 	    ROOTRANK,MPI_COMM_WORLD);
	MPI_Scatter(B, count, MPI_INT, myB, count, MPI_INT,
	                 	    ROOTRANK,MPI_COMM_WORLD);

	/* each processor does a local sum */
	for(int i=0;i<count;i++)
	    myC[i]=myA[i]+myB[i];

	/* send the local sums back to the root */
   MPI_Gather(myC, count, MPI_INT, C, count, MPI_INT, 
	                 	ROOTRANK, MPI_COMM_WORLD);

	/* the root prints the global sum */
	if(myrank == ROOTRANK){
		for(int i=0;i<size;i++)
			printf("%d ",C[i]);
		printf("\n");
	}
   
	MPI_Finalize();
}
