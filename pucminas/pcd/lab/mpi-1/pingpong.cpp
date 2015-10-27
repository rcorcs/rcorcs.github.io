// Ping pong example with MPI_Send and MPI_Recv. Two processes ping pong a
// number back and forth, incrementing it until it reaches a given value.
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
  const int PING_PONG_LIMIT = 10;
  int myrank,nprocs;

  // Initialize the MPI environment
  MPI_Init(&argc,&argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  int ping_pong_count = 0;
  int partner_id = (myrank + 1) % 2;
  while (ping_pong_count < PING_PONG_LIMIT) {
    if (myrank== ping_pong_count % 2) {
      // Increment the ping pong count before you send it
      ping_pong_count++;
      MPI_Send(&ping_pong_count, 1, MPI_INT, partner_id, 0, MPI_COMM_WORLD);
      printf("%d sent and incremented ping_pong_count %d to %d\n",
             myrank, ping_pong_count, partner_id);
    } else {
      MPI_Recv(&ping_pong_count, 1, MPI_INT, partner_id, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      printf("%d received ping_pong_count %d from %d\n",
             myrank, ping_pong_count, partner_id);
    }
  }
  MPI_Finalize();
}
