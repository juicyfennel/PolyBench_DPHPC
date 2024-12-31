#include <stdio.h>
#include <stdlib.h>
#include <time.h> 
#include <assert.h>
#include <mpi.h>

// Problem size
// #define N 30000
#ifndef N 
#define N 12
#endif

// Data type
#define DATA_TYPE double
#define MPI_DATA_TYPE MPI_DOUBLE

#define MALLOC_1D(name, type, nelems) \
 type *name = (type *)xmalloc((size_t)(nelems) * sizeof(type))

// 2D Allocation Macro (1D memory layout to ensure contiguity and efficiency)
#define MALLOC_2D(name, type, nrows, ncols) \
 type *name = (type *)xmalloc((size_t)(nrows) * (size_t)(ncols) * sizeof(type))

// For 2D indexing (if)
#define IDX_2D(name, row, col, ncols) name[(row) * (ncols) + (col)]


static void* xmalloc(size_t alloc_sz)
{
  void* ret = NULL;

  int err = posix_memalign (&ret, 64, alloc_sz);
  if (! ret || err)
    {
      fprintf (stderr, "[PolyBench] posix_memalign: cannot allocate memory");
      exit (1);
    }   

    return ret;
}


void flush_cache()
{
  int cs = 32770 * 1024 * 2 / sizeof(double);
  double* flush = (double*) calloc (cs, sizeof(double));
  int i;
  double tmp = 0.0;
  for (i = 0; i < cs; i++)
    tmp += flush[i];
  assert (tmp <= 10.0);
  free (flush);
}

void init_data(
    DATA_TYPE *alpha,
    DATA_TYPE *beta,
    DATA_TYPE *u1,
    DATA_TYPE *u2, 
    DATA_TYPE *v1,
    DATA_TYPE *v2, 
    DATA_TYPE *y,
    DATA_TYPE *z,
    DATA_TYPE *x,
    DATA_TYPE *w,
    DATA_TYPE *A,
    int start_row,
    int num_rows) {

    DATA_TYPE fn = (DATA_TYPE) N;
    *alpha = 1.5;
    *beta = 1.2;

    int i;

    for (i = 0; i < N; i++)
    {   
        v1[i] = ((i+1)/fn)/4.0;
        v2[i] = ((i+1)/fn)/6.0;
        x[i] = 0.0;
    }
    for(i=start_row; i<start_row+num_rows;i++){
        u1[i-start_row] = i;
        u2[i-start_row] = ((i+1)/fn)/2.0;
        y[i-start_row] = ((i+1)/fn)/8.0;
        z[i-start_row] = ((i+1)/fn)/9.0;
        w[i-start_row] = 0.0;
        for (int j = 0; j < N; j++){
            IDX_2D(A, i-start_row, j, N) = (DATA_TYPE) (i*j % N) / N;
        }
    }
        
}

void kernel_gemver(DATA_TYPE alpha,
    DATA_TYPE beta,
    DATA_TYPE *u1,
    DATA_TYPE *u2, 
    DATA_TYPE *v1,
    DATA_TYPE *v2, 
    DATA_TYPE *y,
    DATA_TYPE *z,
    DATA_TYPE *x,
    DATA_TYPE *w,
    DATA_TYPE *A,
    int start_row,
    int num_rows,
    int rank) {
    int i, j; 

    // Step 3: Every process computes independent rows of A^ 
    for (i = start_row; i < start_row + num_rows; ++i) {
        for (j = 0; j < N; ++j) {
            IDX_2D(A, i-start_row, j, N) = IDX_2D(A, i-start_row, j, N) + u1[i-start_row] * v1[j] + u2[i-start_row] * v2[j];
        }
    }
    


    // Step 4: Every process computes independent components that sum to x together using local rows of A 

    for (i = 0; i < N; i++) {
        for (int j = start_row; j < start_row + num_rows; ++j) {
            x[i] += beta * IDX_2D(A, j-start_row, i, N) * y[j-start_row]; 
        }    
    }

    for (i = start_row; i < start_row+num_rows; i++)
        x[i] = x[i] + z[i-start_row];

    // Step 5: Distribute x to all processes, meanwhile computing their value  
    // MPI_Allreduce(MPI_IN_PLACE, x, N, MPI_DATA_TYPE, MPI_SUM, MPI_COMM_WORLD);

    // MPI_Request request;

    // // Initiate a non-blocking Allreduce operation
    // MPI_Iallreduce(MPI_IN_PLACE, x, N, MPI_DATA_TYPE, MPI_SUM, MPI_COMM_WORLD, &request);

    // // Wait for the operation to complete
    // MPI_Wait(&request, MPI_STATUS_IGNORE);


    // Allocate shared memory for win_buff
    MPI_Win win; 
    double *win_buff;
    MPI_Win_allocate(N * sizeof(double), sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &win_buff, &win);

    // Initialize win_buff to zero on all processes
    for (int i = 0; i < N; i++) {
        win_buff[i] = 0.0;
    }

    // Synchronize before accumulation
    MPI_Win_fence(0, win);

    // Accumulate local x into win_buff using MPI_SUM
    MPI_Accumulate(x, N, MPI_DOUBLE, 0, 0, N, MPI_DOUBLE, MPI_SUM, win);

    // Synchronize to ensure all contributions are completed
    MPI_Win_fence(0, win);

    // Rank 0 now holds the globally summed x in win_buff.
    // Broadcast the result to all processes
    MPI_Bcast(win_buff, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Copy the globally summed result back to x for further computations
    for (int i = 0; i < N; i++) {
        x[i] = win_buff[i];
    }

    // Free the window
    MPI_Win_free(&win);

    // Step 6: Each process computes its portion of w with rows of A 
    for (i = start_row; i < start_row + num_rows; i++) {
        for (j = 0; j < N; j++) {
            w[i - start_row] += alpha * IDX_2D(A, i-start_row, j, N) * x[j];
        }
    }

    if (rank == 1) {
        printf("Own A:\n");
        for (int i = 0; i < num_rows; i++) {
            for (int j = 0; j < N; j++) {
                printf("%f ", IDX_2D(A, i, j, N));
            }
            printf("\n");
        }

        printf("Gathered x:\n");
        for (int i = 0; i < N; i++) {
            printf("%f ", x[i]);
        }
        printf("\n");

        printf("Own w:\n");
        for (int i = 0; i < num_rows; i++) {
            printf("%f ", w[i]);
        }
    }
}


int main(int argc, char** argv) {
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
    DATA_TYPE alpha;
    DATA_TYPE beta; 
    MALLOC_1D(u1, DATA_TYPE, num_rows);
    MALLOC_1D(u2, DATA_TYPE, num_rows);
    MALLOC_1D(v1, DATA_TYPE, N);
    MALLOC_1D(v2, DATA_TYPE, N);
    MALLOC_1D(y, DATA_TYPE, num_rows);
    MALLOC_1D(z, DATA_TYPE, num_rows);
    MALLOC_1D(x, DATA_TYPE, N);
    MALLOC_1D(w, DATA_TYPE, num_rows);
    MALLOC_2D(A, DATA_TYPE, num_rows, N);
    
    /* Initialize array(s). */
    // Step 2: Initialize the local arrays
    init_data(&alpha, &beta, u1, u2, v1, v2, y, z, x, w, A,start_row,num_rows);
    
    // printf("N: %d\n", N);
    // printf("%f", IDX_1D(x, 9));
    
    flush_cache();

    struct timespec start, end; 
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    
    kernel_gemver(alpha, beta, u1, u2, v1, v2, y, z, x, w, A,start_row,num_rows, rank); 

    clock_gettime(CLOCK_MONOTONIC_RAW, &end);

    //printf("Rank %d, Time: %f\n", rank,(end.tv_sec - start.tv_sec) + 1e-9 * (end.tv_nsec - start.tv_nsec));

    // check that A is computed correctly
//    printf("Rows %d - %d, Gathered A:\n", start_row, start_row + num_rows-1);
//    for (int i = 0; i < num_rows; i++) {
//       for (int j = 0; j < n; j++) {
//          printf("%f ", IDX_2D(A,i,j,N));
//       }
//       printf("\n");
//    }

//    // Check that x is computed corectly 
//       if(rank == 0) {
//       printf("Gathered x:\n"); 
//       for (int i = 0; i < n; i++) {
//          printf("%f ", x[i]);
//       }
//       printf("\n");
//    }

//    // check that w is computed correctly
//    printf("Rows %d - %d, Gathered w:\n", start_row, start_row + num_rows-1); 
//    for (int i = 0; i < num_rows; i++) {
//       printf("%f ", w[i]);
//    }
//    printf("\n");

    // Don't forget to free allocated memory
    free(u1);
    free(u2);
    free(v1);
    free(v2);
    free(y);
    free(z);
    free(x);
    free(w);
    free(A);

    MPI_Finalize(); 

    return 0;
}