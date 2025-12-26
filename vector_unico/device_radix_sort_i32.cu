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
  int* vect;
  int* sorted_vector;

  // device vectors
  int* d_vector;
  int* d_sorted_vector;

  // temp variables
  struct timeval tv_start, tv_end;
  double run_time;

  // memmory reserve in CPU
  vect = (int*) malloc(N*sizeof(int));
  sorted_vector = (int*) malloc(N*sizeof(int));

  // memory reserve in GPU
  cudaMalloc(&d_vector, N*sizeof(int));
  cudaMalloc(&d_sorted_vector, N*sizeof(int));

  emptyKernel<<<1, 1>>>(); // GPU warmup

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // Determine temporary device storage requirements
  void* d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;  
  cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_vector, d_sorted_vector, N); 
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);

  // initialization
  srand((unsigned int) time(NULL));
  for(int i = 0; i < N; i++) vect[i]=(rand()%1000);
  
  cudaDeviceSynchronize();
  
  gettimeofday(&tv_start, NULL);

  //device copy
  cudaMemcpyAsync(d_vector, vect, N*sizeof(int), cudaMemcpyHostToDevice, stream);
  // Run sorting operation
  cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_vector, d_sorted_vector, N); 
  // recovery of the parallel results
  cudaMemcpyAsync(sorted_vector, d_sorted_vector, N*sizeof(int), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  gettimeofday(&tv_end, NULL);
  run_time=(tv_end.tv_sec - tv_start.tv_sec) * 1000000 + (tv_end.tv_usec - tv_start.tv_usec); //useconds
  run_time /= 1000000; // seconds
  printf("Tiempo: %lg segundos\n", run_time);

  // free host memory
  free(vect);
  free(sorted_vector);

  // free device memory
  cudaFree(d_vector);
  cudaFree(d_sorted_vector);
  cudaFree(d_temp_storage);
  cudaStreamDestroy(stream);

  return 0;
  
}


