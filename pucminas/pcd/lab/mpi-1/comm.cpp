#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
 
/************************************************************
This is a simple send/receive program in MPI
************************************************************/

#define SOURCE 0
#define DESTINATION 1

int main(int argc, char **argv){
    int myrank, nprocs;
    int num;
 
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);

    if(myrank == SOURCE){
      num=5678;
      MPI_Send(&num,1,MPI_INT,DESTINATION,0,MPI_COMM_WORLD);
      printf("processor %d  sent %d\n",myrank,num);
    }
    if(myrank == DESTINATION){
        MPI_Recv(&num,1,MPI_INT,SOURCE,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        printf("processor %d  got %d\n",myrank,num);
    }
    MPI_Finalize();
}
