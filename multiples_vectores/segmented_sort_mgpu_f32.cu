#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <moderngpu/kernel_segsort.hxx>
#include <moderngpu/context.hxx>

int N = 27000;
int M = 780;

struct less_float {
    __device__ __host__ bool operator()(const float &a, const float &b) const {
        return a < b;
    }
};

using namespace mgpu;

__global__ void emptyKernel() {}

// main function
int main(int argc, char** argv) {

  if(argc > 1) N = atoi(argv[1]);
  if(argc > 1) M = atoi(argv[2]);

  // host vectors
  float* vect;
  float* sorted_vector;

  // device vectors
  float* d_vector;
  float* d_sorted_vector;
  int h_offsets[M+1];
  int* d_offsets;

  // temp variables
  struct timeval tv_start, tv_end;
  double run_time;

  // memmory reserve in CPU
  vect = (float*) malloc(M*N*sizeof(float));
  sorted_vector = (float*) malloc(M*N*sizeof(float));

  // memory reserve in GPU
  cudaMalloc(&d_vector, M*N*sizeof(float));
  cudaMalloc(&d_offsets, (M+1)*sizeof(int));

  emptyKernel<<<1, 1>>>(); // GPU warmup

  for(int i = 0; i < (M+1); i++) h_offsets[i] = N*i;
  cudaMemcpy(d_offsets, h_offsets, (M+1)*sizeof(int), cudaMemcpyHostToDevice);

  // initialization
  srand((unsigned int) time(NULL));
  for(int i = 0; i < M*N; i++) vect[i]=((float)rand()/(float) RAND_MAX);
  
  cudaDeviceSynchronize();
  
  gettimeofday(&tv_start, NULL);

  //device copy
  cudaMemcpy(d_vector, vect, M*N*sizeof(float), cudaMemcpyHostToDevice);
  // Run sorting operation
  mgpu::segmented_sort(d_vector, M*N, d_offsets, M, less_float(), context); 
  // recovery of the parallel results
  cudaMemcpy(sorted_vector, d_vector, M*N*sizeof(float), cudaMemcpyDeviceToHost);

  gettimeofday(&tv_end, NULL);
  run_time=(tv_end.tv_sec - tv_start.tv_sec) * 1000000 + (tv_end.tv_usec - tv_start.tv_usec); //useconds
  run_time /= 1000000; // seconds
  printf("Tiempo: %lg segundos\n", run_time);

  // free host memory
  free(vect);
  free(sorted_vector);

  // free device memory
  cudaFree(d_vector);
  cudaFree(d_offsets);

  return 0;
  
}


