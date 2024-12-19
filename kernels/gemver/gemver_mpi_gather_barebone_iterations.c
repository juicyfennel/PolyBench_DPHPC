#include <stdio.h>
#include <stdlib.h>
#include <time.h> 
#include <assert.h>
#include <mpi.h>

// Problem size
// #define N 30000
#ifndef N 
#define N 25000
#endif

#define ITERATIONS 10

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
    int num_rows) {
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
    MPI_Allreduce(MPI_IN_PLACE, x, N, MPI_DATA_TYPE, MPI_SUM, MPI_COMM_WORLD);

    // printf("Gathered x:\n");
    // for (int i = 0; i < N; i++) {
    //     printf("%f ", x[i]);
    // }
    // printf("\n");

    // Step 6: Each process computes its portion of w with rows of A 
    for (i = start_row; i < start_row + num_rows; i++) {
        for (j = 0; j < N; j++) {
            w[i - start_row] += alpha * IDX_2D(A, i-start_row, j, N) * x[j];
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
    MALLOC_1D(w, DATA_TYPE, rank == 0 ? N : num_rows);
    MALLOC_2D(A, DATA_TYPE, rank == 0 ? N : num_rows, N);

    double total_time = 0.0;

    for (int i=0;i<ITERATIONS;i++){
    
        /* Initialize array(s). */
        // Step 2: Initialize the local arrays
        init_data(&alpha, &beta, u1, u2, v1, v2, y, z, x, w, A,start_row,num_rows);
        
        // printf("N: %d\n", N);
        // printf("%f", IDX_1D(x, 9));
        
        flush_cache();

        struct timespec start, end; 
        clock_gettime(CLOCK_MONOTONIC_RAW, &start);
        
        kernel_gemver(alpha, beta, u1, u2, v1, v2, y, z, x, w, A,start_row,num_rows); 

        clock_gettime(CLOCK_MONOTONIC_RAW, &end);

        total_time += (end.tv_sec - start.tv_sec) + 1e-9 * (end.tv_nsec - start.tv_nsec);


        // Step 7: Gather the computed A and w in rank 0
        clock_gettime(CLOCK_MONOTONIC_RAW, &start);
        MALLOC_1D(sendcounts, int, size);
        MALLOC_1D(displs, int, size);
        int offset = 0;

        //TODO: Gather in size-1 since it should be finished computing rows the earliest
        for(int i = 0; i < size; i++) {
            int rows = rows_per_task + (i < remainder ? 1 : 0);
            sendcounts[i] = rows * n;
            displs[i] = offset;
            offset += rows * n;
        }

        //Gather the results

        if (rank == 0) {
            // Root process uses MPI_IN_PLACE
            MPI_Gatherv(MPI_IN_PLACE, 0, MPI_DATA_TYPE,
                        A, sendcounts, displs, MPI_DATA_TYPE,
                        0, MPI_COMM_WORLD);
        } else {
            // Non-root processes send their portion
            MPI_Gatherv(A, sendcounts[rank], MPI_DATA_TYPE,
                        A, sendcounts, displs, MPI_DATA_TYPE,
                        0, MPI_COMM_WORLD);
        }

        offset = 0; 
        for (int i = 0; i < size; i++) {
            int rows = rows_per_task + (i < remainder ? 1 : 0);
            sendcounts[i] = rows;
            displs[i] = offset;
            offset += rows;
        }
        
        // Gather the results

        if (rank == 0) {
            // Root process uses MPI_IN_PLACE
            MPI_Gatherv(MPI_IN_PLACE, 0, MPI_DATA_TYPE,
                        w, sendcounts, displs, MPI_DATA_TYPE,
                        0, MPI_COMM_WORLD);
        } else {
            // Non-root processes send their portion
            MPI_Gatherv(w, sendcounts[rank], MPI_DATA_TYPE,
                        w, sendcounts, displs, MPI_DATA_TYPE,
                        0, MPI_COMM_WORLD);
        }

        clock_gettime(CLOCK_MONOTONIC_RAW, &end);
        
        total_time += (end.tv_sec - start.tv_sec) + 1e-9 * (end.tv_nsec - start.tv_nsec);
    }
    
    printf("Rank %d, Time: %f\n", rank,total_time);
    



    // check that A is computed correctly
    if (rank == 0){
        printf("Gathered A:\n");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                printf("%f ", IDX_2D(A,i,j,N));
            }
            printf("\n");
        }

        // Check that x is computed corectly 
            
        printf("Gathered x:\n"); 
        for (int i = 0; i < N; i++) {
            printf("%f ", x[i]);
        }
        printf("\n");

        // check that w is computed correctly
        printf("Gathered w:\n"); 
        for (int i = 0; i < N; i++) {
            printf("%f ", w[i]);
        }
        printf("\n");
    }

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