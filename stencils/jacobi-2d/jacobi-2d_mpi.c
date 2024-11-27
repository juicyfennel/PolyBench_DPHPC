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
static void init_array(int n,
                       DATA_TYPE POLYBENCH_2D(A, N, N, n, n),
                       DATA_TYPE POLYBENCH_2D(B, N, N, n, n))
{
  int i, j;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
    {
      A[i][j] = ((DATA_TYPE)i * (j + 2) + 2) / n;
      B[i][j] = ((DATA_TYPE)i * (j + 3) + 3) / n;
    }
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int n,
                        DATA_TYPE POLYBENCH_2D(A, N, N, n, n))

{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("A");
  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
    {
      if ((i * n + j) % 20 == 0)
        fprintf(POLYBENCH_DUMP_TARGET, "\n");
      fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, A[i][j]);
    }
  POLYBENCH_DUMP_END("A");
  POLYBENCH_DUMP_FINISH;
}

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static void kernel_jacobi_2d(int tsteps,
                             int n,
                             DATA_TYPE POLYBENCH_2D(A, N, N, n, n),
                             DATA_TYPE POLYBENCH_2D(B, N, N, n, n),
                             int rank,
                             int size,
                             int col_procs,
                             int row_procs)
{
  int t, i, j;

#pragma scop
  int p_col = rank % col_procs; // This block's col within all blocks
  int p_row = rank / col_procs; // This block's row within all blocks
  int block_length = n / col_procs;
  int block_height = n / row_procs;
  int start_col = p_col * block_length;
  int start_row = p_row * block_height;

  for (t = 0; t < _PB_TSTEPS; t++)
  {
    // Block rows to be sent to other blocks
    DATA_TYPE *s_top_row = &A[start_row][start_col];                       // to be send to upper block as lower buffer
    DATA_TYPE *s_bottom_row = &A[start_row + block_height - 1][start_col]; // to be send to lower block as top buffer

    // Block cols to be sent to other blocks
    DATA_TYPE s_left_col[block_length];
    DATA_TYPE s_right_col[block_length];
    for (int i = start_row; i < start_row + block_height - 1; i++)
    {
      // Iterate over rows i={start_row...(start_row+block_height-1)} for A[start_row + i][...]
      s_left_col[i] = A[i][start_col];
      s_right_col[i] = A[i][start_col + block_length - 1];
    }

    // Block cols and rows to be received from other blocks
    DATA_TYPE *r_top_row    = (DATA_TYPE*)calloc(block_length, sizeof(DATA_TYPE)); // to be received from upper block as top buffer
    DATA_TYPE *r_bottom_row = (DATA_TYPE*)calloc(block_length, sizeof(DATA_TYPE)); // to be received from lower block as lower buffer
    DATA_TYPE *r_left_col   = (DATA_TYPE*)calloc(block_height, sizeof(DATA_TYPE)); // to be received from left block as left buffer
    DATA_TYPE *r_right_col  = (DATA_TYPE*)calloc(block_height, sizeof(DATA_TYPE)); // to be received from right block as right buffer

    // Determine neighbors
    int my_upper_neighbor = (p_row - 1) * col_procs + p_col;
    int my_lower_neighbor = (p_row + 1) * col_procs + p_col;
    int my_left_neighbor  = (p_row * col_procs) + p_col - 1;
    int my_right_neighbor = (p_row * col_procs) + p_col + 1;

    // Check for boundary conditions and set neighbors to MPI_PROC_NULL if they are out of bounds
    if (p_row == 0)           my_upper_neighbor = MPI_PROC_NULL;
    if (p_row == row_procs-1) my_lower_neighbor = MPI_PROC_NULL;
    if (p_col == 0)           my_left_neighbor  = MPI_PROC_NULL;
    if (p_col == col_procs-1) my_right_neighbor = MPI_PROC_NULL;

    // Send and receive data
    MPI_Request mpi_requests[8];
    MPI_Isend(s_top_row,    block_length, MPI_DOUBLE, my_upper_neighbor, 9, MPI_COMM_WORLD, &mpi_requests[0]);
    MPI_Isend(s_bottom_row, block_length, MPI_DOUBLE, my_lower_neighbor, 9, MPI_COMM_WORLD, &mpi_requests[1]);
    MPI_Isend(s_right_col,  block_height, MPI_DOUBLE, my_right_neighbor, 9, MPI_COMM_WORLD, &mpi_requests[2]);
    MPI_Isend(s_left_col,   block_height, MPI_DOUBLE, my_left_neighbor,  9, MPI_COMM_WORLD, &mpi_requests[3]);

    MPI_Irecv(r_top_row,    block_length, MPI_DOUBLE, my_upper_neighbor, 9, MPI_COMM_WORLD, &mpi_requests[4]);
    MPI_Irecv(r_bottom_row, block_length, MPI_DOUBLE, my_lower_neighbor, 9, MPI_COMM_WORLD, &mpi_requests[5]);
    MPI_Irecv(r_right_col,  block_height, MPI_DOUBLE, my_right_neighbor, 9, MPI_COMM_WORLD, &mpi_requests[6]);
    MPI_Irecv(r_left_col,   block_height, MPI_DOUBLE, my_left_neighbor,  9, MPI_COMM_WORLD, &mpi_requests[7]);

    MPI_Waitall(8, mpi_requests, MPI_STATUSES_IGNORE);

    // Update A matrix with received data
    &A[start_row - 1][start_col]            = &r_top_row;
    &A[start_row + block_height][start_col] = &r_bottom_row;
    for (int i = start_row; i < start_row + block_height - 1; i++)
    {
      A[i][start_col - 1]            = r_left_col[i];
      A[i][start_col + block_length] = r_right_col[i];
    }

    // Update B matrix
    for (i = start_row; i < start_row + block_height; i++) {
      for (j = start_col; j < start_col + block_length; j++) {
        B[i][j] = SCALAR_VAL(0.2) * (A[i][j] + A[i][j - 1] + A[i][1 + j] + A[1 + i][j] + A[i - 1][j]);
      }
    }

    for (i = 1; i < _PB_N - 1; i++) {
      for (j = 1; j < _PB_N - 1; j++) {
        A[i][j] = SCALAR_VAL(0.2) * (B[i][j] + B[i][j - 1] + B[i][1 + j] + B[1 + i][j] + B[i - 1][j]);
      }
    }
  }
#pragma endscop
}

int main(int argc, char **argv)
{
  /* Retrieve problem size. */
  int n = N;
  int tsteps = TSTEPS;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);
  POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, N, N, n, n);

  /* MPI vars */
  int rank, size;

  /* Initialize MPI */
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  /* TODO: calculate number of row and col processes */

  /* TODO: initialize arrays (remember to add 2 extra  rows and cols to accomodate overlap) */

  /* Initialize array(s). */
  init_array(n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_jacobi_2d(tsteps, n, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), rank, size);

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(A)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);

  return 0;
}
