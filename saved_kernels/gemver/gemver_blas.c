#include <stdio.h>
#include <stdlib.h>
#include <time.h> 
#include <assert.h>
#include <cblas.h> // Include OpenBLAS

// Problem size
#ifndef PROBLEM_SIZE
#define PROBLEM_SIZE 25000
#endif

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
  int err = posix_memalign (&ret, 4096, alloc_sz);
  if (!ret || err)
  {
      fprintf (stderr, "[PolyBench] posix_memalign: cannot allocate memory");
      exit (1);
  }   
  return ret;
}

void flush_cache()
{
  int cs = 32770 * 1024 * 2 / sizeof(double);
  double* flush = (double*) calloc(cs, sizeof(double));
  int i;
  double tmp = 0.0;
  for (i = 0; i < cs; i++)
    tmp += flush[i];
  assert(tmp <= 10.0);
  free(flush);
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
    int N) 
{
    DATA_TYPE fn = (DATA_TYPE) N;
    *alpha = 1.5;
    *beta = 1.2;

    #pragma omp parallel for
    for (int i = 0; i < N; i++) {   
        u1[i] = i;
        u2[i] = ((i + 1) / fn) / 2.0;
        v1[i] = ((i + 1) / fn) / 4.0;
        v2[i] = ((i + 1) / fn) / 6.0;
        y[i] = ((i + 1) / fn) / 8.0;
        z[i] = ((i + 1) / fn) / 9.0;
        x[i] = 0.0;
        w[i] = 0.0;

        for (int j = 0; j < N; j++) {
            A[i * N + j] = (DATA_TYPE) (i * j % N) / N; 
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
                   int N) 
{
    // 1) A^ = A + u1 * v1^T + u2 * v2^T
    // dger multiplies vector X by the transpose of vector Y, then adds matrix A (double precison).
    cblas_dger(CblasRowMajor, N, N, 1.0, u1, 1, v1, 1, A, N); 
    cblas_dger(CblasRowMajor, N, N, 1.0, u2, 1, v2, 1, A, N);


    // 2) x = beta * (A^)^T * y + z
    // dgemv multiplies a matrix by a vector (double precision). This calculates x <- beta * (A^)^T * y + 1 * x
    // Here, have to transpose A^
    cblas_dgemv(CblasRowMajor, CblasTrans, N, N, beta, A, N, y, 1, 1.0, x, 1);
    // Now, x = beta * (A^)^T * y and we have to add z 
    // daxpy computes a constant times a vector plus a vector (double-precision) and places result in second vector 
    cblas_daxpy(N, 1.0, z, 1, x, 1); 

    // 3) w = alpha * A^ * x 
    // dgemv multiplies a matrix by a vector (double precision). This calculates w <- alpha * A^ * x + 1 * w
    // don't transpose A
    cblas_dgemv(CblasRowMajor, CblasNoTrans, N, N, alpha, A, N, x, 1, 0.0, w, 1);

//         printf("Gathered A_hat:\n");
//     for (int i = 0; i < N; i++) {
//         for (int j = 0; j < N; j++) {
//             printf("%f ", IDX_2D(A, i, j, N));
//       }
//       printf("\n");
//    }

    // printf("Gathered x:\n");
    // for (int i = 0; i < N; i++) {
    //     printf("%f ", x[i]);
    // }
    // printf("\n");

    // printf("Gathered w:\n");
    // for (int i = 0; i < N; i++) {
    //     printf("%f ", w[i]);
    // }
    // printf("\n");
}

int main(int argc, char** argv) 
{
    int N = PROBLEM_SIZE;
    // omp_set_num_threads(4); // Set 4 threads for OpenMP
    DATA_TYPE alpha, beta; 
    MALLOC_1D(u1, DATA_TYPE, N);
    MALLOC_1D(u2, DATA_TYPE, N);
    MALLOC_1D(v1, DATA_TYPE, N);
    MALLOC_1D(v2, DATA_TYPE, N);
    MALLOC_1D(y, DATA_TYPE, N);
    MALLOC_1D(z, DATA_TYPE, N);
    MALLOC_1D(x, DATA_TYPE, N);
    MALLOC_1D(w, DATA_TYPE, N);
    MALLOC_2D(A, DATA_TYPE, N, N);

    printf("N: %d\n", N);
    double total_time = 0;

    flush_cache();
    init_data(&alpha, &beta, u1, u2, v1, v2, y, z, x, w, A, N);

    struct timespec start, end; 
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    kernel_gemver(alpha, beta, u1, u2, v1, v2, y, z, x, w, A, N); 
    clock_gettime(CLOCK_MONOTONIC_RAW, &end);

    printf("Time: %f\n", (end.tv_sec - start.tv_sec) + 1e-9 * (end.tv_nsec - start.tv_nsec));

    free(u1); free(u2); free(v1); free(v2); 
    free(y); free(z); free(x); free(w); free(A);
    return 0;
}
