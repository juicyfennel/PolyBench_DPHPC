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
static void init_array(int n, // Size of the total matrix
                       int start_row, // Start row of the block
                       int start_col, // Start col of the block
                       int block_height, // Height of the block
                       int block_length, // Length of the block
                       DATA_TYPE POLYBENCH_2D(A, block_height + 2, block_length + 2, block_height + 2, block_length + 2), // Block A
                       DATA_TYPE POLYBENCH_2D(B, block_height + 2, block_length + 2, block_height + 2, block_length + 2)) // Block B
{
  // Iterate over all elements of the block with a padding of 1
  for (int i = 0; i < block_height + 2; i++) {
    for (int j = 0; j < block_length + 2; j++) {
      A[i][j] = ((DATA_TYPE)(start_row + i) * (start_col + j + 2) + 2) / n;
      B[i][j] = ((DATA_TYPE)(start_row + i) * (start_col + j + 3) + 3) / n;
    }
  }
}

static void init_res_array(int n,
                           DATA_TYPE POLYBENCH_2D(A_res, N, N, n, n)) // Block A_res
{
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      A_res[i][j] = ((DATA_TYPE)(i) * (j + 2) + 2) / n;
    }
  }
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_inner_array(int rank, // Rank of the process
                        int block_height, // Height of the block
                        int block_length, // Length of the block
                        DATA_TYPE POLYBENCH_2D(A, block_height+2, block_length+2, block_height+2, block_length+2))

{
  POLYBENCH_DUMP_START;
  fprintf(POLYBENCH_DUMP_TARGET, "Rank %d\n", rank);
  POLYBENCH_DUMP_BEGIN("A\n");


  for (int i = 1; i < block_height + 1; i++) {
    for (int j = 1; j < block_length + 1; j++) {
      fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, A[i][j]);
    }
    fprintf(POLYBENCH_DUMP_TARGET, "\n");
  }
  POLYBENCH_DUMP_END("A");
  POLYBENCH_DUMP_FINISH;
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_outer_array(int rank, // Rank of the process
                        int block_height, // Height of the block
                        int block_length, // Length of the block
                        DATA_TYPE POLYBENCH_2D(A, block_height + 2, block_length + 2, block_height + 2, block_length + 2))

{
  POLYBENCH_DUMP_START;
  fprintf(POLYBENCH_DUMP_TARGET, "Rank %d\n", rank);
  POLYBENCH_DUMP_BEGIN("A\n");


  for (int i = 0; i < block_height + 2; i++) {
    for (int j = 0; j < block_length + 2; j++) {
      fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, A[i][j]);
    }
    fprintf(POLYBENCH_DUMP_TARGET, "\n");
  }
  POLYBENCH_DUMP_END("A");
  POLYBENCH_DUMP_FINISH;
}

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static void kernel_jacobi_2d(int tsteps,
                             int block_height, // Height of the block
                             int block_length, // Length of the block
                             int p_row, // This block's row within all blocks
                             int p_col, // This block's col within all blocks
                             DATA_TYPE POLYBENCH_2D(A, block_height+2, block_length+2, block_height+2, block_length+2),
                             DATA_TYPE POLYBENCH_2D(B, block_height+2, block_length+2, block_height+2, block_length+2),
                             int rank,
                             int size,
                             MPI_Comm cart_comm)
{
  int t, i, j;

#pragma scop

  // Create MPI Datatypes for rows and columns
  MPI_Datatype row_type, col_type;
  MPI_Type_contiguous(block_length, MPI_DOUBLE, &row_type);
  MPI_Type_commit(&row_type);
  MPI_Type_vector(block_height, 1, block_length + 2, MPI_DOUBLE, &col_type);
  MPI_Type_commit(&col_type);

  int coords[2];
  int up, down, left, right;
  MPI_Cart_coords(cart_comm, rank, 2, coords);

  // Determine neighbors
  MPI_Cart_shift(cart_comm, 0, 1, &up, &down);
  MPI_Cart_shift(cart_comm, 1, 1, &left, &right);

  MPI_Request mpi_requests[8];

  for (t = 0; t < _PB_TSTEPS; t++) {

    // Send and receive data
    MPI_Isend(&A[1][1],            1, row_type, up,    0, cart_comm, &mpi_requests[0]);
    MPI_Isend(&A[block_height][1], 1, row_type, down,  0, cart_comm, &mpi_requests[1]);
    MPI_Isend(&A[1][block_length], 1, col_type, right, 0, cart_comm, &mpi_requests[2]);
    MPI_Isend(&A[1][1],            1, col_type, left,  0, cart_comm, &mpi_requests[3]);

    MPI_Irecv(&A[0][1],              1, row_type, up,    0, cart_comm, &mpi_requests[4]);
    MPI_Irecv(&A[block_height+1][1], 1, row_type, down,  0, cart_comm, &mpi_requests[5]);
    MPI_Irecv(&A[1][block_length+1], 1, col_type, right, 0, cart_comm, &mpi_requests[6]);
    MPI_Irecv(&A[1][0],              1, col_type, left,  0, cart_comm, &mpi_requests[7]);

    MPI_Waitall(8, mpi_requests, MPI_STATUSES_IGNORE);

    // Update B matrix
    for (i = 1; i < block_height + 1; i++) {
      for (j = 1; j < block_length + 1; j++) {
        B[i][j] = SCALAR_VAL(0.2) * (A[i][j] + A[i][j - 1] + A[i][1 + j] + A[1 + i][j] + A[i - 1][j]);
      }
    }

    MPI_Isend(&B[1][1],            1, row_type, up,    0, cart_comm, &mpi_requests[0]);
    MPI_Isend(&B[block_height][1], 1, row_type, down , 0, cart_comm, &mpi_requests[1]);
    MPI_Isend(&B[1][block_length], 1, col_type, right, 0, cart_comm, &mpi_requests[2]);
    MPI_Isend(&B[1][1],            1, col_type, left,  0, cart_comm, &mpi_requests[3]);

    MPI_Irecv(&B[0][1],              1, row_type, up,    0, cart_comm, &mpi_requests[4]);
    MPI_Irecv(&B[block_height+1][1], 1, row_type, down,  0, cart_comm, &mpi_requests[5]);
    MPI_Irecv(&B[1][block_length+1], 1, col_type, right, 0, cart_comm, &mpi_requests[6]);
    MPI_Irecv(&B[1][0],              1, col_type, left,  0, cart_comm, &mpi_requests[7]);

    MPI_Waitall(8, mpi_requests, MPI_STATUSES_IGNORE);

    for (i = 1; i < block_height + 1; i++) {
      for (j = 1; j < block_length + 1; j++) {
        A[i][j] = SCALAR_VAL(0.2) * (B[i][j] + B[i][j - 1] + B[i][1 + j] + B[1 + i][j] + B[i - 1][j]);
      }
    }
  }
  
  MPI_Type_free(&row_type);
  MPI_Type_free(&col_type);

#pragma endscop
}



/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_res_array(int n,
                        DATA_TYPE POLYBENCH_2D(A_res, N, N, n, n))

{
  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("A\n");


  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, A_res[i][j]);
    }
    fprintf(POLYBENCH_DUMP_TARGET, "\n");
  }
  POLYBENCH_DUMP_END("A");
  POLYBENCH_DUMP_FINISH;
}

int main(int argc, char **argv)
{
  /* Retrieve problem size. */
  int n = N;
  int tsteps = TSTEPS;

  /* MPI vars */
  int rank, size;

  /* Initialize MPI */
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int dims[2] = {0, 0};
  int periods[2] = {0, 0};
  MPI_Dims_create(size, 2, dims);
  MPI_Comm cart_comm;
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);

  /* Calculate number of row and col processes */

  int p_row = rank / dims[1]; // This block's row within all blocks
  int p_col = rank % dims[1]; // This block's col within all blocks
  int block_height = (n - 2) / dims[0];
  int block_length = (n - 2) / dims[1];
  int start_row = p_row * block_height;
  int start_col = p_col * block_length;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(A,
                          DATA_TYPE,
                          block_height + 2,
                          block_length + 2,
                          block_height + 2,
                          block_length + 2);
  POLYBENCH_2D_ARRAY_DECL(B,
                          DATA_TYPE,
                          block_height + 2,
                          block_length + 2,
                          block_height + 2,
                          block_length + 2);

  /* Initialize array(s). */
  init_array(n,
             start_row,
             start_col,
             block_height,
             block_length,
             POLYBENCH_ARRAY(A),
             POLYBENCH_ARRAY(B));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_jacobi_2d(tsteps,
                   block_height,
                   block_length,
                   p_row,
                   p_col,
                   POLYBENCH_ARRAY(A),
                   POLYBENCH_ARRAY(B),
                   rank,
                   size,
                   cart_comm);

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  // // Gather all data in rank 0

  double (*A_res)[N][N];
  int *displs = NULL;
  int *recv_counts = NULL;

  if (rank == 0) {
    A_res = (double(*)[N][N])malloc((N) * (N) * sizeof(double));

    init_res_array(n,
              POLYBENCH_ARRAY(A_res));

    recv_counts = malloc(size * sizeof(int));
    displs = malloc(size * sizeof(int));

    // Calculate displacements for each block
    for (int i = 0; i < size; i++) {
        int proc_coords[2];
        MPI_Cart_coords(cart_comm, i, 2, proc_coords);
        int global_start_row = proc_coords[0] * block_height + 1;
        int global_start_col = proc_coords[1] * block_length + 1;
        displs[i] = global_start_row * N + global_start_col; // Flattened index for subarray
        recv_counts[i] = 1; // Each process sends one inner block
    }
  }

  int coords[2];
  MPI_Cart_coords(cart_comm, rank, 2, coords);

  MPI_Datatype block_type, res_block_type;
  MPI_Type_vector(block_height, block_length, block_length + 2, MPI_DOUBLE, &block_type);
  MPI_Type_commit(&block_type);

  int sizes[2] = {N, N}; // Size of the global matrix
  int subsizes[2] = {block_height, block_length}; // Size of the block
  int starts[2] = {coords[0] * block_height + 1, coords[1] * block_length + 1}; // Starting position of the block
  MPI_Type_create_subarray(2, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &res_block_type);
  MPI_Type_commit(&res_block_type);

  print_outer_array(rank, block_height, block_length, POLYBENCH_ARRAY(A));

  // printf("A[1][1]: %f\n", *A[1][1]);

  MPI_Gatherv(&A[1][1],
              1,
              block_type,
              rank == 0 ? A_res : NULL,
              rank == 0 ? recv_counts : NULL,
              rank == 0 ? displs : NULL,
              res_block_type,
              0,
              MPI_COMM_WORLD);

  
  
  if (rank == 0) {
    polybench_prevent_dce(print_res_array(n, POLYBENCH_ARRAY(A_res)));
    free(A_res);
  }


  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);

  // MPI_Type_free(&block_type);
  // MPI_Type_free(&res_block_type);

  MPI_Finalize();

  return 0;
}
