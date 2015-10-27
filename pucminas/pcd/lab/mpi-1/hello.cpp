#include <mpi.h>
#include <stdio.h>
 
int main (int argc, char* argv[]){
    int myrank, nprocs;
    MPI_Init(&argc, &argv); //starts MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank); //get current process id
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs); //get number of processes
    printf("Hello world from process %d of %d\n", myrank, nprocs );

    MPI_Finalize();
    return 0;
}
