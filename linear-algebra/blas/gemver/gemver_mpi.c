#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "gemver.h"


/* Array initialization. */
static
void init_array (int n,
                 DATA_TYPE *alpha,
                 DATA_TYPE *beta,
                 DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
DATA_TYPE POLYBENCH_1D(u1,N,n),
DATA_TYPE POLYBENCH_1D(v1,N,n),
DATA_TYPE POLYBENCH_1D(u2,N,n),
DATA_TYPE POLYBENCH_1D(v2,N,n),
DATA_TYPE POLYBENCH_1D(w,N,n),
DATA_TYPE POLYBENCH_1D(x,N,n),
DATA_TYPE POLYBENCH_1D(y,N,n),
DATA_TYPE POLYBENCH_1D(z,N,n))
{
int i, j;

*alpha = 1.5;
*beta = 1.2;

DATA_TYPE fn = (DATA_TYPE)n;

for (i = 0; i < n; i++)
{
u1[i] = i;
u2[i] = ((i+1)/fn)/2.0;
v1[i] = ((i+1)/fn)/4.0;
v2[i] = ((i+1)/fn)/6.0;
y[i] = ((i+1)/fn)/8.0;
z[i] = ((i+1)/fn)/9.0;
x[i] = 0.0;
w[i] = 0.0;
for (j = 0; j < n; j++)
A[i][j] = (DATA_TYPE) (i*j % n) / n;
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
                   DATA_TYPE POLYBENCH_2D(A,N,N,n,n),
DATA_TYPE POLYBENCH_1D(u1,N,n),
DATA_TYPE POLYBENCH_1D(v1,N,n),
DATA_TYPE POLYBENCH_1D(u2,N,n),
DATA_TYPE POLYBENCH_1D(v2,N,n),
DATA_TYPE POLYBENCH_1D(w,N,n),
DATA_TYPE POLYBENCH_1D(x,N,n),
DATA_TYPE POLYBENCH_1D(y,N,n),
DATA_TYPE POLYBENCH_1D(z,N,n),
int rank, 
int size)
{
int i, j;

#pragma scop   

// TODO: Do we have to do initial scatterv to distribute initial rows of A? I.e. not have all data on all processes. Ask Alexandru 
// TODO: Check if gathering everything in the final process (which does less computations, but has less data) is faster than in the first one 
// TODO: Check whether gathering in-place (everywhere) is faster than 1D/2D allocations

// Step 0: Compute which rows of A this process is responsible for 
int rows_per_task = n / size; // Floor 
int remainder = n % size;

int start_row = rank * rows_per_task + (rank < remainder ? rank : remainder);
int num_rows = rows_per_task + (rank < remainder ? 1 : 0);

// Step 1: Every process computes independent rows of A^ 
for (i = start_row; i < start_row + num_rows; ++i) {
   for (j = 0; j < n; ++j) {
      A[i][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j];
   }
}

// Step 2: Gather the computed A_Hat in rank 0 -- finished with first step 
int* sendcounts = malloc(size * sizeof(int));
int* displs = malloc(size * sizeof(int));
int offset = 0;

// TODO: Gather in size-1 since it should be finished computing rows the earliest 
for (i = 0; i < size; i++) {
   int rows = rows_per_task + (i < remainder ? 1 : 0);
   sendcounts[i] = rows * n;
   displs[i] = offset;
   offset += rows * n;
}

// Gather the results

if (rank == 0) {
    // Root process uses MPI_IN_PLACE
    MPI_Gatherv(MPI_IN_PLACE, 0, MPI_DATA_TYPE,
                &A[0][0], sendcounts, displs, MPI_DATA_TYPE,
                0, MPI_COMM_WORLD);
} else {
    // Non-root processes send their portion
    MPI_Gatherv(&A[start_row][0], sendcounts[rank], MPI_DATA_TYPE,
                &A[0][0], sendcounts, displs, MPI_DATA_TYPE,
                0, MPI_COMM_WORLD);
}

// if (rank == 0) {
//    printf("Gathered A_hat:\n");
//     for (int i = 0; i < n; i++) {
//         for (int j = 0; j < n; j++) {
//             printf("%f ", A[i][j]);
//             // printf("%f ", A_hat[i*n+j]);
//       }
//       printf("\n");
//    }
// }

// Step 3: Every process computes independent components that sum to x together using local rows of A 
DATA_TYPE* local_x = (DATA_TYPE*)calloc(n, sizeof(DATA_TYPE));

for (i = 0; i < n; i++) {
   for (int j = start_row; j < start_row + num_rows; ++j) {
      local_x[i] += beta * A[j][i] * y[j]; 
   }
}

for (int i = start_row; i < start_row+num_rows; ++i) {
   local_x[i] += z[i];
}

// Step 4: Distribute x to all processes, meanwhile computing their value  
MPI_Allreduce(local_x, x, n, MPI_DATA_TYPE, MPI_SUM, MPI_COMM_WORLD);

// Check that x is computed corectly 
// if (rank == 0) {
//    printf("Gathered x:\n"); 
//    for (i = 0; i < n; i++) {
//       printf("%f ", x[i]);
//    }
//    printf("\n");
// }

// Step 5: Each process computes its portion of w with rows of A 
DATA_TYPE* local_w = (DATA_TYPE*)calloc(n, sizeof(DATA_TYPE));
for (i = start_row; i < start_row + num_rows; i++) {
   for (j = 0; j < n; j++) {
      local_w[i - start_row] += alpha * A[i][j] * x[j];
   }
}

// Step 6: Gather the results of w into process 0
MPI_Gather(local_w, num_rows, MPI_DATA_TYPE,
         w, num_rows, MPI_DATA_TYPE,
         0, MPI_COMM_WORLD);

// if (rank == 0) {
//    printf("Gathered w:\n"); 
//       for (i = 0; i < n; i++) {
//          printf("%f ", w[i]);
//       }
//    printf("\n");
// }

free(sendcounts);
free(displs);
free(local_w);
free(local_x);

#pragma endscop
}


int main(int argc, char** argv)
{
    /* Retrieve problem size. */
    int n = N;

    /* Variable declaration/allocation. */
    DATA_TYPE alpha;
    DATA_TYPE beta;
    POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);
    POLYBENCH_1D_ARRAY_DECL(u1, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(v1, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(u2, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(v2, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(w, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(y, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(z, DATA_TYPE, N, n);


    /* Initialize array(s). */
    init_array (n, &alpha, &beta,
                POLYBENCH_ARRAY(A),
                POLYBENCH_ARRAY(u1),
                POLYBENCH_ARRAY(v1),
                POLYBENCH_ARRAY(u2),
                POLYBENCH_ARRAY(v2),
                POLYBENCH_ARRAY(w),
                POLYBENCH_ARRAY(x),
                POLYBENCH_ARRAY(y),
                POLYBENCH_ARRAY(z));

   // MPI vars 
   int rank, size;

    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

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
                   rank, size);

    /* Stop and print timer. */
    polybench_stop_instruments;

    // ONLY FOR RANK 0 PRINT 
    if (rank == 0) polybench_print_instruments;
    

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

    MPI_Finalize(); 

    return 0;
}