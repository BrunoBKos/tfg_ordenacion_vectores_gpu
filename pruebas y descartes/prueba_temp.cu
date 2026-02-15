#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <cub/cub.cuh>

int N = 1024;

__global__ void emptyKernel() {}

// main function
int main(int argc, char** argv) {

  if(argc > 1) N = atoi(argv[1]);

  // host vectors
  float* vect;
  float* sorted_vector;

  // device vectors
  float* d_input;
  float* d_output;

  // temp variables
  struct timeval tv_start, tv_end;
  struct timeval tv_start_2, tv_end_2;
  double run_time;

  // memmory reserve in CPU
  vect = (float*) malloc(N*sizeof(float));
  sorted_vector = (float*) malloc(N*sizeof(float));

  // memory reserve in GPU
  cudaMalloc(&d_input, N*sizeof(float));
  cudaMalloc(&d_output, N*sizeof(float));

  emptyKernel<<<1, 1>>>(); // GPU warmup

  // initialization
  srand((unsigned int) time(NULL));
  for(int i = 0; i < N; i++) vect[i]=((float)rand()/(float) RAND_MAX);

  // Determine temporary device storage requirements
  void* d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;

  cudaMemcpy(d_input, vect, N*sizeof(float), cudaMemcpyHostToDevice);

  cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_input, d_output, N);

  cudaMemcpy(sorted_vector, d_output, N*sizeof(float), cudaMemcpyDeviceToHost);

  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);
  
  gettimeofday(&tv_start, NULL);

  //device copy
  cudaMemcpy(d_input, vect, N*sizeof(float), cudaMemcpyHostToDevice);

  // Run sorting operation
  gettimeofday(&tv_start_2, NULL);
  cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_input, d_output, N);   
  cudaDeviceSynchronize();
  gettimeofday(&tv_end_2, NULL);

  // recovery of the parallel results
  cudaMemcpy(sorted_vector, d_output, N*sizeof(float), cudaMemcpyDeviceToHost);

  gettimeofday(&tv_end, NULL);
  run_time=(tv_end.tv_sec - tv_start.tv_sec) * 1000000 + (tv_end.tv_usec - tv_start.tv_usec); //useconds
  run_time /= 1000000; // seconds
  printf("Ejecucion total\n");
  printf("Tiempo: %lg segundos\n", run_time);
  printf("-----------------------------------\n");
  run_time=(tv_end_2.tv_sec - tv_start_2.tv_sec) * 1000000 + (tv_end_2.tv_usec - tv_start_2.tv_usec); //useconds
  run_time /= 1000000; // seconds
  printf("Ejecucion parcial\n");
  printf("Tiempo: %lg segundos\n", run_time);
  printf("-----------------------------------\n");

  // free host memory
  free(vect);
  free(sorted_vector);

  // free device memory
  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_temp_storage);

  return 0;
  
}


