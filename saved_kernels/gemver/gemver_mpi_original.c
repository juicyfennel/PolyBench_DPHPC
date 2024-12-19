#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "gemver.h"

// #define N 12
/* Array initialization. */
static
void init_array (int n,
                 DATA_TYPE *alpha,
                 DATA_TYPE *beta,
                 DATA_TYPE A[][N],
DATA_TYPE *u1,
DATA_TYPE *v1,
DATA_TYPE *u2,
DATA_TYPE *v2,
DATA_TYPE *w,
DATA_TYPE *x,
DATA_TYPE *y,
DATA_TYPE *z,
int start_row,
int num_rows)
{
int i, j;

*alpha = 1.5;
*beta = 1.2;

DATA_TYPE fn = (DATA_TYPE)n;

for (i = 0; i < n; i++)
{
v1[i] = ((i+1)/fn)/4.0;
v2[i] = ((i+1)/fn)/6.0;
x[i] = 0.0;
}
for(i=start_row;i<start_row+num_rows;i++){
   u1[i-start_row] = i;
   u2[i-start_row] = ((i+1)/fn)/2.0;
   y[i-start_row] = ((i+1)/fn)/8.0;
   z[i-start_row] = ((i+1)/fn)/9.0;
   w[i-start_row] = 0.0;
   for (j = 0; j < n; j++)
      A[i-start_row][j] = (DATA_TYPE) (i*j % n) / n;
}
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n,
                 DATA_TYPE POLYBENCH_1D(w,N,n))
{
int i;

POLYBENCH_DUMP_START;
POLYBENCH_DUMP_BEGIN("w");
for (i = 0; i < n; i++) {
if (i % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, w[i]);
}
POLYBENCH_DUMP_END("w");
POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_gemver(int n,
                 DATA_TYPE alpha,
                 DATA_TYPE beta,
                 DATA_TYPE A[][N],
DATA_TYPE *u1,
DATA_TYPE *v1,
DATA_TYPE *u2,
DATA_TYPE *v2,
DATA_TYPE *w,
DATA_TYPE *x,
DATA_TYPE *y,
DATA_TYPE *z,
int start_row, 
int num_rows)
{
int i, j;

#pragma scop   

// DONE: Do we have to do initial scatterv to distribute initial rows of A? I.e. not have all data on all processes. Ask Alexandru 
// TODO: Check if gathering everything in the final process (which does less computations, but has less data) is faster than in the first one 
// TODO: Check whether gathering in-place (everywhere) is faster than 1D/2D allocations (we care only for x)

// Step 3: Every process computes independent rows of A^ 
for (i = start_row; i < start_row + num_rows; ++i) {
   for (j = 0; j < n; ++j) {
      A[i-start_row][j] = A[i-start_row][j] + u1[i-start_row] * v1[j] + u2[i-start_row] * v2[j];
   }
}

// Step 4: Every process computes independent components that sum to x together using local rows of A 

for (i = 0; i < n; i++) {
   for (int j = start_row; j < start_row + num_rows; ++j) {
      x[i] += beta * A[j-start_row][i] * y[j-start_row]; 
   }
}

for (int i = start_row; i < start_row+num_rows; ++i) {
   x[i] += z[i-start_row];
}

// Step 5: Distribute x to all processes, meanwhile computing their value  
MPI_Allreduce(MPI_IN_PLACE, x, n, MPI_DATA_TYPE, MPI_SUM, MPI_COMM_WORLD);

// Check that x is computed corectly 
// if (rank == 0) {
//    printf("Gathered x:\n"); 
//    for (i = 0; i < n; i++) {
//       printf("%f ", x[i]);
//    }
//    printf("\n");
// }

// Step 6: Each process computes its portion of w with rows of A 
for (i = start_row; i < start_row + num_rows; i++) {
   for (j = 0; j < n; j++) {
      w[i - start_row] += alpha * A[i-start_row][j] * x[j];
   }
}

#pragma endscop
}


int main(int argc, char** argv)
{
   /* Retrieve problem size. */
   int n = N;

   // MPI vars 
   int rank, size;

   // Initialize MPI
   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &size);

   
   // Step 0: Compute which rows this process is responsible for 
   int rows_per_task = n / size; // Floor 
   int remainder = n % size;
   int start_row = rank * rows_per_task + (rank < remainder ? rank : remainder);
   int num_rows = rows_per_task + (rank < remainder ? 1 : 0);

   /* Variable declaration/allocation. */
   // Step 1: Allocate the local arrays (everything now is local)
   DATA_TYPE alpha;
   DATA_TYPE beta;
   POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, rank == 0 ? N : num_rows, N, n, n); // only dim1 and dim2 are used
   POLYBENCH_1D_ARRAY_DECL(u1, DATA_TYPE, num_rows, n);
   POLYBENCH_1D_ARRAY_DECL(v1, DATA_TYPE, N, n);
   POLYBENCH_1D_ARRAY_DECL(u2, DATA_TYPE, num_rows, n);
   POLYBENCH_1D_ARRAY_DECL(v2, DATA_TYPE, N, n);
   POLYBENCH_1D_ARRAY_DECL(w, DATA_TYPE, rank == 0 ? N : num_rows, n);
   POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE, N, n);
   POLYBENCH_1D_ARRAY_DECL(y, DATA_TYPE, num_rows, n);
   POLYBENCH_1D_ARRAY_DECL(z, DATA_TYPE, num_rows, n);


   /* Initialize array(s). */
   // Step 2: Initialize the local arrays
   init_array (n, &alpha, &beta,
               POLYBENCH_ARRAY(A),
               POLYBENCH_ARRAY(u1),
               POLYBENCH_ARRAY(v1),
               POLYBENCH_ARRAY(u2),
               POLYBENCH_ARRAY(v2),
               POLYBENCH_ARRAY(w),
               POLYBENCH_ARRAY(x),
               POLYBENCH_ARRAY(y),
               POLYBENCH_ARRAY(z),
               start_row, num_rows);
   
   /* Start timer. */
   polybench_start_instruments;

   /* Run kernel. */
   kernel_gemver (n, alpha, beta,
                  POLYBENCH_ARRAY(A),
                  POLYBENCH_ARRAY(u1),
                  POLYBENCH_ARRAY(v1),
                  POLYBENCH_ARRAY(u2),
                  POLYBENCH_ARRAY(v2),
                  POLYBENCH_ARRAY(w),
                  POLYBENCH_ARRAY(x),
                  POLYBENCH_ARRAY(y),
                  POLYBENCH_ARRAY(z),
                  start_row, num_rows);

   /* Stop and print timer. */
   polybench_stop_instruments;

   printf("Rank %d, Time for Kernel calculation: ", rank);
   polybench_print_instruments;
   // Step 7: Gather the computed A and w in rank 0
   polybench_start_instruments;

   POLYBENCH_1D_ARRAY_DECL(sendcounts, int, size, size);
   POLYBENCH_1D_ARRAY_DECL(displs, int, size, size);
   int offset = 0;

   // TODO: Gather in size-1 since it should be finished computing rows the earliest 
   for (int i = 0; i < size; i++) {
      int rows = rows_per_task + (i < remainder ? 1 : 0);
      (POLYBENCH_ARRAY(sendcounts))[i] = rows * n;
      (POLYBENCH_ARRAY(displs))[i] = offset;
      offset += rows * n;
   }

   // Gather the results

   if (rank == 0) {
      // Root process uses MPI_IN_PLACE
      MPI_Gatherv(MPI_IN_PLACE, 0, MPI_DATA_TYPE,
                  POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(sendcounts), POLYBENCH_ARRAY(displs), MPI_DATA_TYPE,
                  0, MPI_COMM_WORLD);
   } else {
      // Non-root processes send their portion
      MPI_Gatherv(POLYBENCH_ARRAY(A), (POLYBENCH_ARRAY(sendcounts))[rank], MPI_DATA_TYPE,
                  POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(sendcounts), POLYBENCH_ARRAY(displs), MPI_DATA_TYPE,
                  0, MPI_COMM_WORLD);
   }

   offset = 0;
   for (int i = 0; i < size; i++) {
      int rows = rows_per_task + (i < remainder ? 1 : 0);
      (POLYBENCH_ARRAY(sendcounts))[i] = rows;
      (POLYBENCH_ARRAY(displs))[i] = offset;
      offset += rows;
   }

   // Gather the results

   if (rank == 0) {
      // Root process uses MPI_IN_PLACE
      MPI_Gatherv(MPI_IN_PLACE, 0, MPI_DATA_TYPE,
                  &(POLYBENCH_ARRAY(w))[0], POLYBENCH_ARRAY(sendcounts), POLYBENCH_ARRAY(displs), MPI_DATA_TYPE,
                  0, MPI_COMM_WORLD);
   } else {
      // Non-root processes send their portion
      MPI_Gatherv(&(POLYBENCH_ARRAY(w))[0], (POLYBENCH_ARRAY(sendcounts))[rank], MPI_DATA_TYPE,
                  &(POLYBENCH_ARRAY(w))[0], POLYBENCH_ARRAY(sendcounts), POLYBENCH_ARRAY(displs), MPI_DATA_TYPE,
                  0, MPI_COMM_WORLD);
   }

   polybench_stop_instruments;

   //ONLY FOR RANK 0 PRINT 
   if (rank == 0) {
      printf("Time for Gather: ");
      polybench_print_instruments;
   }

   // if (rank == 0) {
   //    printf("Gathered A:\n");
   //    for (int i = 0; i < n; i++) {
   //       for (int j = 0; j < n; j++) {
   //          printf("%f ", (POLYBENCH_ARRAY(A))[i][j]);
   //       }
   //       printf("\n");
   //    }
   
   //    // Check that x is computed corectly 
   //    printf("Gathered x:\n"); 
   //    for (int i = 0; i < n; i++) {
   //       printf("%f ", (*x)[i]);
   //    }
   //    printf("\n");

   //    // check that w is computed correctly
   //    printf("Gathered w:\n"); 
   //    for (int i = 0; i < n; i++) {
   //       printf("%f ", (*w)[i]);
   //    }
   //    printf("\n");
   // }
   
   
   

   /* Prevent dead-code elimination. All live-out data must be printed
      by the function call in argument. */
   polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(w)));

   /* Be clean. */
   POLYBENCH_FREE_ARRAY(A);
   POLYBENCH_FREE_ARRAY(u1);
   POLYBENCH_FREE_ARRAY(v1);
   POLYBENCH_FREE_ARRAY(u2);
   POLYBENCH_FREE_ARRAY(v2);
   POLYBENCH_FREE_ARRAY(w);
   POLYBENCH_FREE_ARRAY(x);
   POLYBENCH_FREE_ARRAY(y);
   POLYBENCH_FREE_ARRAY(z);
   POLYBENCH_FREE_ARRAY(sendcounts);
   POLYBENCH_FREE_ARRAY(displs);

   MPI_Finalize(); 

   return 0;
}