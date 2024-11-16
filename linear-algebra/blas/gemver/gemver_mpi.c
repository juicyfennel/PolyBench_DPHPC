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

// // Declare as 2D array with num_rows_per_process rows and N columns
// POLYBENCH_2D_ARRAY_DECL(local_A_hat, DATA_TYPE, num_rows_per_process, N, num_rows_per_process, n);

// Step 1: Compute local rows of A^
// Every process processes floor(n/p) rows sequentially, e.g. if N=10 and P=3, then P0 processes rows 0-3, P1 4-7, P2 8,9
// Could also do ceil: #rows_per_process = (n+size-1)/size; if end_row > n: end_row = n
// To balance better, could also do cyclically, but harder to distribute after 
// TODO: can't we just do this with scattering A from rank 0 process? --> TODO: IMPLEMENT LIKE THIS AND COMPARE? 

// Initially, only run with num 
// Has to be divisible amount of processes
int num_rows_per_process = n / size;
int start_row = rank * num_rows_per_process;
int end_row = start_row + num_rows_per_process;

// Ensure contiguous in memory -- 2D array seems to be good 
// Note row_i(A_hat) = row_i(A) + (u_1)_i * v_1 + (u_2)_i * v_2
DATA_TYPE local_A_hat[num_rows_per_process][n];
for (i = start_row; i < end_row; i++) {
   // printf("rank %i calculated row %i of A^\n", rank, i);
   for (j = 0; j < n; j++) {
      // local_A_hat[(i - start_row)*n+j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j];
      local_A_hat[i - start_row][j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j];

   }
}

// // Step 2: Gather A^ into process 0 TODO: Don't receive into 0, but into n-1? One that does least work depending on whether evenly divides
// // https://www.mpich.org/static/docs/v3.3/www3/MPI_Gather.html
// May have to replace by explicitly contiguous version through malloc but seems to work
// DATA_TYPE A_hat[n][n]; // Only meaningful in process 0
// DATA_TYPE *A_hat = (DATA_TYPE *)malloc(n * n * sizeof(DATA_TYPE));

// For non-perfect divisible n/size. Automatically truncates, but should actually replace with Gatherv since variable length data from different processes
// Technically A_hat[n][n] wouldn't fit, e.g. if we have 3 processes for N=10, ceil(N/P) = 4, so we actually get 12 rows of data
// For now, this seems to work though
MPI_Gather(local_A_hat, num_rows_per_process * n, MPI_DOUBLE,
         A, num_rows_per_process * n, MPI_DOUBLE,
         0, MPI_COMM_WORLD);


// Step 3: Redistribute columns of A^ into all processes 
// CANT SEEM TO GET THIS TO WORK -- otherwise could just scatter etc.
// For column dtype: https://stackoverflow.com/questions/10788180/sending-columns-of-a-matrix-using-mpi-scatter
// int num_cols_per_process = n/size;  // Assume this divides evenly 
// MPI_Datatype col, coltype;
// MPI_Type_vector(N, 1, N, MPI_DOUBLE, &col);
// MPI_Type_commit(&col);
// MPI_Type_create_resized(col, 0, 1*sizeof(double), &coltype); // Used for the scatter so it knows where next column is 
// MPI_Type_commit(&coltype);

// DATA_TYPE local_A_hat_cols[num_cols_per_process][n];

// // // Could reuse num_rows_per_process, since NxN 
// // // Scatter divides columns evenly, first N/P to P0, second N/P to P1, etc.
// MPI_Scatter(A_hat, num_cols_per_process, coltype,
//             local_A_hat_cols, num_cols_per_process, coltype,
//             0, MPI_COMM_WORLD);

// // After using the derived datatypes
// MPI_Type_free(&col);
// MPI_Type_free(&coltype);

// Instead: Broadcast the full A_hat matrix to all processes
// Ideally scatter columns instead
MPI_Bcast(A, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

// Step 4: Calculate x components given the correct cols of A
// Note x_i = beta * col_i(A_hat)^T y + z_i 
int num_cols_per_process = n / size;
int start_col = rank * num_cols_per_process;
int end_col = end_col + num_cols_per_process;

DATA_TYPE local_x[num_cols_per_process];

for (i = start_row; i < end_row; i++) {
   local_x[i-start_row] = 0;
   for (int j = 0; j < n; ++j) {
      local_x[i-start_row] += beta * A[j][i] * y[j]; 
   }

   local_x[i-start_row] += z[i];
}


// Now, 2 choices
// Note that we can interpret 
// w = beta * A_hat * x 
// as 
// 1) beta * (x_1 * col_1(A) + x_2 * col_2(A) + ... + x_n * col_N(A))
// Now, (this approach might be even better if each process also has only columns of A_hat, rather than full A_hat)
// we can choose to only calculate part of w with the x_i that we have locally, so we don't have to re-gather and re-broadcast the full x
// (ideally also don't have to regather and rebroadcast A_hat in this case)
// Then, at the end, reduce all 'local w' calculations by summing them up in a single process --> but this takes a lot of computation

// Instead, 
// 2) beta * [row_1(A)^Tx row_2(A)^Tx ... row_n(A)^x]^T 
// In this case, first need to locally gather x in some process, then redistribute it, then let each process compute individidual 
// components of w based on the rows of A that it has 
// So, more independent computation on the processes (no final reduce), but also more communication
// Went for option 2 for now 
MPI_Gather(local_x, num_cols_per_process, MPI_DOUBLE,
         x, num_rows_per_process, MPI_DOUBLE,
         0, MPI_COMM_WORLD);

// Step 5: Broadcast x to all processes
MPI_Bcast(x, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

// Step 5: Each process computes its portion of w

DATA_TYPE local_w[num_rows_per_process];
for (i = start_row; i < end_row; i++) {
   // printf("rank %i calculated row %i of A^\n", rank, i);
   for (j = 0; j < n; j++) {
      // local_A_hat[(i - start_row)*n+j] = A[i][j] + u1[i] * v1[j] + u2[i] * v2[j];
      local_w[i - start_row] += alpha * A[i][j] * x[j];
   }
}



// Step 6: Gather the results of w into process 0
MPI_Gather(local_w, num_rows_per_process, MPI_DOUBLE,
         w, num_rows_per_process, MPI_DOUBLE,
         0, MPI_COMM_WORLD);



// if (rank == 0) {
//    printf("Gathered A_hat:\n");
//     for (int i = 0; i < n; i++) {
//         for (int j = 0; j < n; j++) {
//             printf("%f ", A[i][j]);
//             // printf("%f ", A_hat[i*n+j]);
//       }
//       printf("\n");
//    }

//    printf("Gathered x:\n"); 
//    for (i = 0; i < n; i++) {
//       printf("%f ", x[i]);
//    }
//    printf("\n");

//    printf("Gathered w:\n"); 
//    for (i = 0; i < n; i++) {
//       printf("%f ", w[i]);
//    }
//    printf("\n");

   // printf("Local A^ rows of rank 0 (first n/size):\n");
   // for (i = 0; i < n/size; i++) {
   //    for (j = 0; j < n; j++) {
   //          printf("%f ", local_A_hat[i][j]);
   //          // printf("%f ", local_A_hat[i*n+j]);
   //    }
   //    printf("\n");
   // }

   // printf("Local elems of x of rank 0 (first n/size):\n");
   // for (i = 0; i < n/size; i++) {
   //    printf("%f ", local_x[i]);
   // }
   // printf("\n");

   // printf("Local elems of w of rank 0 (first n/size):\n");
   // for (i = 0; i < n/size; i++) {
   //    printf("%f ", local_w[i]);
   // }
   // printf("\n");

   // printf("Local A^ cols of rank 0 (first n/size):\n");
   // for (i = 0; i < n/size; i++) {
   //    for (j = 0; j < n; j++) {
   //          printf("%f ", local_A_hat_cols[i][j]);
   //    }
   //    printf("\n");
   // }
// }

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