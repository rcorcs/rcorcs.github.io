#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <mpi.h>

#define SOURCE 0

int main(int argc, char **argv){
    int myrank, nprocs;
    int num = 0;
    if(argc==2)
		num = atoi(argv[1]);

    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    
    int split = ceil(float(num)/nprocs);
    int begin = split*myrank;
    int end = split*(myrank+1);
    if(begin==0){
       begin = 1;
    }
    if(end>=num){
       end = num;
    }
    printf("split: %d; begin: %d; end: %d\n", split, begin, end);

    int localsum = 0;
    for(int div = begin; div<end; div++){
        if(num%div==0){
             localsum += div;
        }
    }

   printf("processor %d has sum %d\n", myrank, localsum);

   if(myrank == SOURCE){
      int sum = localsum;
      int tmp;
      for(int i = 1; i<nprocs; i++){
         MPI_Recv(&tmp, 1, MPI_INT, i, 0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
         sum += tmp;
         printf("received from processor %d, %d, %d\n", i, tmp, sum);
      }

      printf("results from all processors: %d\n",sum);
   }else{
      MPI_Send(&localsum,1,MPI_INT,SOURCE,0,MPI_COMM_WORLD);
   }

   MPI_Finalize();
}






