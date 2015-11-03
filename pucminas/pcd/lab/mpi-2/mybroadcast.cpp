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
    
   if(pid==source) {
      // If we are the root process, send our data to everyone
      for(int i = 0; i < nprocs; i++){
         if(i!=pid){
            MPI_Send(&num, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
            printf("process %d sent %f to process %d\n",pid,num,i);
         }
      }
   }else{
      // If we are a receiver process, receive the data from the root
      MPI_Recv(&num, 1, MPI_DOUBLE, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      printf("process %d received %f\n",pid,num);
   }

   MPI_Finalize();
}

