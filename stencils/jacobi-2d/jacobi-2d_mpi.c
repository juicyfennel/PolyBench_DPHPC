#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "jacobi-2d.h"

#define DEBUG 1

/* Array initialization. */
static void init_array(int n, // Size of the matrix without padding
                       int start_row, // Start row of the block
                       int start_col, // Start col of the block
                       int block_height, // Height of the block
                       int block_length, // Length of the block
                       DATA_TYPE POLYBENCH_2D(A, N, N, n, n), // Block A
                       DATA_TYPE POLYBENCH_2D(B, N, N, n, n)) // Block B
{
  // Iterate over all elements of the block with a padding of 1
  for (int i = 1; i <= block_height; i++) {
    for (int j = 1; j <= block_length; j++) {
      A[i][j] = ((DATA_TYPE)(start_row + i -1) * (start_col + j + 1) + 2) / n;
      B[i][j] = ((DATA_TYPE)(start_row + i -1) * (start_col + j + 2) + 3) / n;
    }
  }
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int block_height, // Height of the block
                        int block_length, // Length of the block
                        DATA_TYPE POLYBENCH_2D(A, N, N, n, n))

{
  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("A");

  for (int i = 0; i < block_height + 1; i++) {
    for (int j = 0; j < block_length + 1; j++) {
      fprintf(POLYBENCH_DUMP_TARGET, "A[%d][%d] = %f\n", i, j, A[i][j]);
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
                             DATA_TYPE POLYBENCH_2D(A, N, N, n, n),
                             DATA_TYPE POLYBENCH_2D(B, N, N, n, n),
                             int rank,
                             int size,
                             int col_procs,
                             int row_procs)
{
  int t, i, j;

  #if DEBUG
  printf("Rank %d has entered kernel_jacobi_2d with arguments: tsteps=%d, block_height=%d, block_length=%d, p_row=%d, p_col=%d, rank=%d, size=%d, col_procs=%d, row_procs=%d.\n", rank, tsteps, block_height, block_length, p_row, p_col, rank, size, col_procs, row_procs);
  #endif

#pragma scop
  for (t = 0; t < 1; t++) {
    // Block rows to be sent to other blocks
    // Notice indices 1 and {block_height - 2} are used to avoid sending the padding
    DATA_TYPE *s_top_row = A[1];                   // to be send to upper block as lower buffer
    DATA_TYPE *s_bottom_row = A[block_height - 2]; // to be send to lower block as top buffer
    #if DEBUG
    printf("Rank %d has copied s_top_row and s_bottom_row.\n", rank);
    #endif

    // Block cols to be sent to other blocks
    // Notice indices 1 and {block_length - 2} are used to avoid sending the padding
    DATA_TYPE s_left_col[block_height];
    DATA_TYPE s_right_col[block_height];
    #if DEBUG
    printf("Rank %d has initialized s_left_col and s_right_col.\n", rank);
    #endif
    for (int i = 0; i < block_height; i++) {
      // Notice indices 1 and block_length - 2 are used to avoid sending the padding
      s_left_col[i] = A[i][1];
      s_right_col[i] = A[i][block_length - 2];
    }
    #if DEBUG
    printf("Rank %d has copied s_left_col and s_right_col.\n", rank);
    #endif

    #if DEBUG
    MPI_Barrier(MPI_COMM_WORLD);
    printf("SYNC-2: Rank %d has synchronized.\n", rank);
    #endif
    
    // Block cols and rows to be received from other blocks
    DATA_TYPE *r_top_row    = (DATA_TYPE*)calloc(block_length, sizeof(DATA_TYPE)); // to be received from upper block as top buffer
    DATA_TYPE *r_bottom_row = (DATA_TYPE*)calloc(block_length, sizeof(DATA_TYPE)); // to be received from lower block as lower buffer
    DATA_TYPE *r_left_col   = (DATA_TYPE*)calloc(block_height, sizeof(DATA_TYPE)); // to be received from left block as left buffer
    DATA_TYPE *r_right_col  = (DATA_TYPE*)calloc(block_height, sizeof(DATA_TYPE)); // to be received from right block as right buffer
    #if DEBUG
    printf("Rank %d has calloced all its receiving arrays..\n", rank);
    #endif

    #if DEBUG
    MPI_Barrier(MPI_COMM_WORLD);
    printf("SYNC-3: Rank %d has synchronized.\n", rank);
    #endif

    // // Determine neighbors
    int my_upper_neighbor = (p_row - 1) * col_procs + p_col;
    int my_lower_neighbor = (p_row + 1) * col_procs + p_col;
    int my_left_neighbor  = (p_row * col_procs) + p_col - 1;
    int my_right_neighbor = (p_row * col_procs) + p_col + 1;

    // Check for boundary conditions and set neighbors to MPI_PROC_NULL if they are out of bounds
    if (p_row == 0)           my_upper_neighbor = MPI_PROC_NULL;
    if (p_row == row_procs-1) my_lower_neighbor = MPI_PROC_NULL;
    if (p_col == 0)           my_left_neighbor  = MPI_PROC_NULL;
    if (p_col == col_procs-1) my_right_neighbor = MPI_PROC_NULL;

    #if DEBUG
    MPI_Barrier(MPI_COMM_WORLD);
    printf("SYNC-4: Rank %d has synchronized.\n", rank);
    printf("Rank %d: p_row: %d\n", rank, p_row);
    printf("Rank %d: p_col: %d\n", rank, p_col);
    printf("Rank %d: my_upper_neighbor: %d\n", rank, my_upper_neighbor);
    printf("Rank %d: my_lower_neighbor: %d\n", rank, my_lower_neighbor);
    printf("Rank %d: my_left_neighbor: %d\n", rank, my_left_neighbor);
    printf("Rank %d: my_right_neighbor: %d\n", rank, my_right_neighbor);
    #endif

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

    #if DEBUG
    MPI_Barrier(MPI_COMM_WORLD);
    printf("SYNC-5: Rank %d has synchronized.\n", rank);
    // for (int i = 0; i < block_length; i++) {
    //   printf("Rank %d: r_top_row[%d] = %f\n", rank, i, r_top_row[i]);
    //   printf("Rank %d: r_bottom_row[%d] = %f\n", rank, i, r_bottom_row[i]);
    // }
    // for (int i = 0; i < block_height; i++) {
    //   printf("Rank %d: r_left_col[%d] = %f\n", rank, i, r_left_col[i]);
    //   printf("Rank %d: r_right_col[%d] = %f\n", rank, i, r_right_col[i]);
    // }
    #endif

    // Update A matrix with received data
    memcpy(A[0], r_top_row, block_length * sizeof(DATA_TYPE));
    memcpy(A[block_height - 1], r_bottom_row, block_length * sizeof(DATA_TYPE));
    for (int i = 0; i < block_height; i++)
    {
      A[i][0]                = r_left_col[i];
      A[i][block_length - 1] = r_right_col[i];
    }
    #if DEBUG
    MPI_Barrier(MPI_COMM_WORLD);
    printf("SYNC-6: Rank %d has synchronized.\n", rank);
    #endif

    // Update B matrix
    for (i = 1; i < block_height - 1; i++) {
      for (j = 1; j < block_length - 1; j++) {
        B[i][j] = SCALAR_VAL(0.2) * (A[i][j] + A[i][j - 1] + A[i][1 + j] + A[1 + i][j] + A[i - 1][j]);
      }
    }

    #if DEBUG
    MPI_Barrier(MPI_COMM_WORLD);
    printf("SYNC-7: Rank %d has synchronized.\n", rank);
    #endif

    for (i = 1; i < block_height - 1; i++) {
      for (j = 1; j < block_length - 1; j++) {
        A[i][j] = SCALAR_VAL(0.2) * (B[i][j] + B[i][j - 1] + B[i][1 + j] + B[1 + i][j] + B[i - 1][j]);
      }
    }
    // Do a synchronization after each iteration
    MPI_Barrier(MPI_COMM_WORLD);

    #if DEBUG
    printf("SYNC-8: Rank %d has synchronized.\n", rank);
    #endif
  }
#pragma endscop
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

  #if DEBUG
  printf("Rank %d of %d initialized.\n", rank, size);
  #endif

  /* Calculate number of row and col processes */
  int col_procs = sqrt(size); // Assuming a square grid of processes
  int row_procs = size / col_procs;

  int p_col = rank % col_procs; // This block's col within all blocks
  int p_row = rank / col_procs; // This block's row within all blocks
  int block_length = n / col_procs;
  int block_height = n / row_procs;
  int start_col = p_col * block_length;
  int start_row = p_row * block_height;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, block_height + 2, block_length + 2);
  POLYBENCH_2D_ARRAY_DECL(B, DATA_TYPE, N, N, block_height + 2, block_length + 2);

  #if DEBUG
  printf("Rank %d declared A and B matrices of size %d*%d.\n", rank, block_height + 2, block_length + 2);
  printf("Rank %d will call init_array with n=%d, start_row=%d, start_col=%d, block_height=%d, block_length=%d.\n", rank, n, start_row, start_col, block_height, block_length);
  #endif

  /* Initialize array(s). */
  init_array(n,
             start_row,
             start_col,
             block_height,
             block_length,
             POLYBENCH_ARRAY(A),
             POLYBENCH_ARRAY(B));

  #if DEBUG
  printf("Rank %d initialized A and B matrices of size %d*%d.\n", rank, block_height + 2, block_length + 2);
  #endif

  /* Synchronize all processes */
  MPI_Barrier(MPI_COMM_WORLD);
  #if DEBUG
  printf("SYNC-1: Rank %d has synchronized.\n", rank);
  #endif

  /* Start timer. */
  polybench_start_instruments;
  #if DEBUG
  printf("Rank %d has called polybench_start_instruments.\n", rank);
  #endif

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
                   col_procs,
                   row_procs);

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;
  #if DEBUG
  printf("Rank %d has called polybench_stop_instruments.\n", rank);
  #endif

  double *final_A = NULL;
  if (rank == 0) {
    final_A = (double*)calloc(n * n, sizeof(double)); // Full matrix on root
  }
  
  // MPI_Gather Parameters
  int *sendcounts = malloc(size * sizeof(int)); // Number of elements to send to each process
  int sendcount_for_each_process = block_height * block_length;
  int *displacements = malloc(size * sizeof(int));     // Displacement of each block in the receive buffer
  for (int i = 0; i < size; i++) {
    sendcounts[i] = sendcount_for_each_process; // Number of elements in each block
    if (i == 0) {
      displacements[i] = 0; // Displacement of each block
    } else {
      displacements[i] = displacements[i - 1] + sendcounts[i - 1]; // Displacement of each block
    }
  }

  #if DEBUG
  printf("Rank %d has calculated sendcounts and displs.\n", rank);
  if (rank == 0) {
    printf("Rank %d: sendcounts = [", rank);
    for (int i = 0; i < size - 1; i++) {
      printf("%d, ", sendcounts[i]);
    }
    printf("%d]\n", sendcounts[size - 1]);
    printf("Rank %d: displs = [", rank);
    for (int i = 0; i < size - 1; i++) {
      printf("%d, ", displacements[i]);
    }
    printf("%d]\n", displacements[size - 1]);
  }
  #endif

  // Gather the blocks from all processes
  

  // Comment by Alexandre: The following line is commented out because it results in a segfault.
  // I'm suggesting we analyze the MPI_Gatherv function to understand why it's causing the segfault.
  // For example with a tool like *valgrind*, which is very powerful but not Apple Silicon compatible.

  // MPI_Gatherv(final_A,
  //             block_height * block_length,
  //             MPI_DOUBLE,
  //             final_A,
  //             sendcounts,
  //             displacements,
  //             MPI_DOUBLE,
  //             0,
  //             MPI_COMM_WORLD);

  // Print the final matrix
  if (rank == 0) {
    #if DEBUG
    // print_array(n, n, final_A);
    #endif
    free(final_A);
  }

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, n, POLYBENCH_ARRAY(A)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);

  #if DEBUG
  printf("Rank %d says \"au revoir les amis\".\n", rank);
  #endif
  MPI_Finalize();

  return 0;
}
