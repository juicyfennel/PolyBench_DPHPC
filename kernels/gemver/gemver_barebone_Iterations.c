#include <stdio.h>
#include <stdlib.h>
#include <time.h> 
#include <assert.h>

// Problem size
// #define N 30000
#ifndef N 
#define N 25000
#endif

#define ITERATIONS 10

// Data type
#define DATA_TYPE double

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
    DATA_TYPE *A) {

    DATA_TYPE fn = (DATA_TYPE) N;
    *alpha = 1.5;
    *beta = 1.2;

    for (int i = 0; i < N; i++)
    {   
        u1[i] = i;
        u2[i] = ((i+1)/fn)/2.0;
        v1[i] = ((i+1)/fn)/4.0;
        v2[i] = ((i+1)/fn)/6.0;
        y[i] = ((i+1)/fn)/8.0;
        z[i] = ((i+1)/fn)/9.0;
        x[i] = 0.0;
        w[i] = 0.0;

        for (int j = 0; j < N; j++)
            IDX_2D(A, i, j, N) = (DATA_TYPE) (i*j % N) / N;
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
    DATA_TYPE *A) {
    int i, j; 

    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
        //A[i * N + j] = A[i * N + j] + u1[i] * v1[j] + u2[i] * v2[j];
        IDX_2D(A, i, j, N) = IDX_2D(A, i, j, N) + u1[i] * v1[j] + u2[i] * v2[j];
    
//     printf("Gathered A_hat:\n");
//     for (int i = 0; i < N; i++) {
//         for (int j = 0; j < N; j++) {
//             printf("%f ", IDX_2D(A, i, j, N));
//       }
//       printf("\n");
//    }

    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
            x[i] = x[i] + beta * IDX_2D(A, j, i, N)*y[j];


    for (i = 0; i < N; i++)
        x[i] = x[i] + z[i];

    // printf("Gathered x:\n");
    // for (int i = 0; i < N; i++) {
    //     printf("%f ", x[i]);
    // }
    // printf("\n");

    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
        w[i] = w[i] +  alpha * IDX_2D(A, i, j, N) * x[j];

    // printf("Gathered w:\n");
    // for (int i = 0; i < N; i++) {
    //     printf("%f ", w[i]);
    // }
    // printf("\n");
}


int main(int argc, char** argv) {
    /* Variable declaration/allocation. */
    DATA_TYPE alpha;
    DATA_TYPE beta; 
    MALLOC_1D(u1, DATA_TYPE, N);
    MALLOC_1D(u2, DATA_TYPE, N);
    MALLOC_1D(v1, DATA_TYPE, N);
    MALLOC_1D(v2, DATA_TYPE, N);
    MALLOC_1D(y, DATA_TYPE, N);
    MALLOC_1D(z, DATA_TYPE, N);
    MALLOC_1D(x, DATA_TYPE, N);
    MALLOC_1D(w, DATA_TYPE, N);
    MALLOC_2D(A, DATA_TYPE, N, N);
    
    init_data(&alpha, &beta, u1, u2, v1, v2, y, z, x, w, A);
    
    printf("N: %d\n", N);
    // printf("%f", IDX_1D(x, 9));

    //compute total time
    double total_time = 0.0;

    for (int i = 0; i < ITERATIONS; i++) {
        init_data(&alpha, &beta, u1, u2, v1, v2, y, z, x, w, A);
        flush_cache();

        struct timespec start, end; 
        clock_gettime(CLOCK_MONOTONIC_RAW, &start);
        
        kernel_gemver(alpha, beta, u1, u2, v1, v2, y, z, x, w, A); 

        clock_gettime(CLOCK_MONOTONIC_RAW, &end);
        total_time += (end.tv_sec - start.tv_sec) + 1e-9 * (end.tv_nsec - start.tv_nsec);
    }

    printf("Time: %f\n", total_time);

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
    
    return 0;
}