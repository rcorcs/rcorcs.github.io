#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>
 
/************************************************************
This is a simple broadcast program in MPI
************************************************************/

int main(int argc, char **argv){
    int pid, nprocs;
    int source;
    double num;
 
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&pid);
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    source=0;
    if(pid==source){
        num=3.1415;
    }
    MPI_Bcast(&num,1,MPI_DOUBLE,source,MPI_COMM_WORLD);
    printf("process %d received %f\n",pid,num);
    MPI_Finalize();
}

