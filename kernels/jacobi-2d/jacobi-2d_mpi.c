#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "jacobi-2d.h"


/* Array initialization. */
static
void init_array (int n,
		 DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
		 DATA_TYPE POLYBENCH_2D(B,N,N,n,n))
{
  int i, j;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      {
	A[i][j] = ((DATA_TYPE) i*(j+2) + 2) / n;
	B[i][j] = ((DATA_TYPE) i*(j+3) + 3) / n;
      }
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
		 DATA_TYPE POLYBENCH_2D(A,N,N,n,n))

{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("A");
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++) {
      if ((i * n + j) % 20 == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, A[i][j]);
    }
  POLYBENCH_DUMP_END("A");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_jacobi_2d(int tsteps,
			    int n,
			    DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
			    DATA_TYPE POLYBENCH_2D(B,N,N,n,n))
{
  int t, i, j;

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int n_rows = N / size;
  int start = rank * n_rows + 1;
  int end = start + n_rows - 1;

#pragma scop
  MPI_Request send_requests[2], recv_requests[2];

  for (t = 0; t < TSTEPS; t++) {
      // Exchange boundary rows with neighbors for B matrix update
      if (rank > 0) { // Send top row and receive from above
          MPI_Isend(&A[start][0], N, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &send_requests[0]);
          MPI_Irecv(&A[start - 1][0], N, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &recv_requests[0]);
      }
      if (rank < size - 1) { // Send bottom row and receive from below
          MPI_Isend(&A[end][0], N, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &send_requests[1]);
          MPI_Irecv(&A[end + 1][0], N, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &recv_requests[1]);
      }

      // Wait for all communications to complete
      if (rank > 0) MPI_Wait(&recv_requests[0], MPI_STATUS_IGNORE);
      if (rank < size - 1) MPI_Wait(&recv_requests[1], MPI_STATUS_IGNORE);

      // Compute B matrix
      for (i = start; i <= end; i++) {
          for (j = 1; j < N - 1; j++) {
              B[i][j] = SCALAR_VAL(0.2) * (A[i][j] + A[i][j - 1] + A[i][j + 1] + A[i + 1][j] + A[i - 1][j]);
          }
      }

      // Exchange boundary rows with neighbors for A matrix update
      if (rank > 0) {
          MPI_Isend(&B[start][0], N, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &send_requests[0]);
          MPI_Irecv(&B[start - 1][0], N, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &recv_requests[0]);
      }
      if (rank < size - 1) {
          MPI_Isend(&B[end][0], N, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &send_requests[1]);
          MPI_Irecv(&B[end + 1][0], N, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &recv_requests[1]);
      }

      // Wait for all communications to complete
      if (rank > 0) MPI_Wait(&recv_requests[0], MPI_STATUS_IGNORE);
      if (rank < size - 1) MPI_Wait(&recv_requests[1], MPI_STATUS_IGNORE);

      // Compute A matrix
      for (i = start; i <= end; i++) {
          for (j = 1; j < N - 1; j++) {
              A[i][j] = SCALAR_VAL(0.2) * (B[i][j] + B[i][j - 1] + B[i][j + 1] + B[i + 1][j] + B[i - 1][j]);
          }
      }
  }
#pragma endscop

}


int main(int argc, char** argv)
{
  /* Init MPI */
  MPI_Init(&argc, &argv);

  /* Retrieve problem size. */
  int n = N;
  int tsteps = TSTEPS;

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);
  POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, N, N, n, n);

  /* Initialize array(s). */
  init_array (n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

  /* Start timer. */
  if (rank == 0)
    polybench_start_instruments;

  /* Run kernel. */
  kernel_jacobi_2d(tsteps, n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

  /* Stop and print timer. */
  if (rank == 0) {
    polybench_stop_instruments;
    polybench_print_instruments;

    /* Prevent dead-code elimination. All live-out data must be printed
      by the function call in argument. */
    polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(A)));
  }
  
  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);

  /* Finalize MPI */
  MPI_Finalize();

  return 0;
}
