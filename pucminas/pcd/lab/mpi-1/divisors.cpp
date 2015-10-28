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
    int num = 0;
    if(argc==2)
		num = atoi(argv[1]);
    printf("argc: %d; num: %d\n", argc, num);

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

   printf("%d\n", localsum);
   int sum;
   MPI_Reduce(&localsum, &sum, 1, MPI_INT, MPI_SUM, SOURCE, MPI_COMM_WORLD);
   
   // the root prints the global sum
   if(myrank == SOURCE){
      printf("results from all processors= %d\n",sum);
   }

   MPI_Finalize();
}
