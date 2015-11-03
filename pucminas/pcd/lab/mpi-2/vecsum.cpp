#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

/*
! This program shows how to use MPI_Scatter and MPI_Reduce
! Each processor gets different data from the root processor
! by way of mpi_scatter.  The data is summed and then sent back
! to the root processor using MPI_Reduce.  The root processor
! then prints the global sum. 
*/

#define ROOTRANK 0

int main(int argc,char **argv){
	int myrank, nprocs;

   int *myarr,*arr;
   int count;
   int gtotal;
	
   MPI_Init(&argc,&argv);
   MPI_Comm_size( MPI_COMM_WORLD, &nprocs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

   // each processor will get count elements from the root
   count=2;
   myarr=(int*)malloc(count*sizeof(int));
   // create the data to be sent on the root
   if(myrank == ROOTRANK){
      int size=count*nprocs;
      arr=(int*)malloc(size*sizeof(int));
      for(int i=0;i<size;i++)
         arr[i]=i+1;
   }
   // send different data to each processor
   MPI_Scatter(arr, count, MPI_INT, myarr,count,MPI_INT,
                         ROOTRANK,MPI_COMM_WORLD);

   // each processor does a local sum
   int total=0;
   for(int i=0;i<count;i++)
      total = total+myarr[i];

   printf("myrank= %d total= %d\n",myrank,total);

   // send the local sums back to the root
   MPI_Reduce(&total, &gtotal, 1, MPI_INT, MPI_SUM,
                         ROOTRANK, MPI_COMM_WORLD);
   // the root prints the global sum
   if(myrank == ROOTRANK){
      printf("results from all processors: %d\n",gtotal);
   }

   MPI_Finalize();
   return 0;
}

