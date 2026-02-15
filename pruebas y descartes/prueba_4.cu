#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <cub/cub.cuh>

int N = 27400;
int M = 10;

__global__ void emptyKernel() {}

// main function
int main(int argc, char** argv) {

  if(argc > 1) M = atoi(argv[1]);
  if(argc > 2) N = atoi(argv[2]);

  // host vectors
  short* vect;
  short* sorted_vector;

  // device vectors
  short* d_vector;
  short* d_sorted_vector;

  // temp variables
  struct timeval tv_start, tv_end;
  double run_time;

  // memmory reserve in CPU
  vect = (short*) malloc(M*N*sizeof(short));
  sorted_vector = (short*) malloc(M*N*sizeof(short));

  // memory reserve in GPU
  cudaMalloc(&d_vector, N*sizeof(short));
  cudaMalloc(&d_sorted_vector, N*sizeof(short));

  emptyKernel<<<1, 1>>>(); // GPU warmup

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // Determine temporary device storage requirements
  void* d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;  
  cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_vector, d_sorted_vector, N); 
  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);

  for(int i = 0; i < 11; i++) {
    // initialization
    srand((unsigned int) time(NULL));
    for(int i = 0; i < N*M; i++) vect[i]= (short) (rand()%1000);
    
    cudaDeviceSynchronize();

    gettimeofday(&tv_start, NULL);

    short* vect_aux = vect;
    short* sorted_vector_aux = sorted_vector;
    for(int j = 0; j < M; j++) {
        //device copy
        cudaMemcpyAsync(d_vector, vect_aux, N*sizeof(short), cudaMemcpyHostToDevice, stream);
        // Run sorting operation
        cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_vector, d_sorted_vector, N); 
        // recovery of the parallel results
        cudaMemcpyAsync(sorted_vector_aux, d_sorted_vector, N*sizeof(short), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        vect_aux += N;
        sorted_vector_aux += N;
    }
    gettimeofday(&tv_end, NULL);
    run_time=(tv_end.tv_sec - tv_start.tv_sec) * 1000000 + (tv_end.tv_usec - tv_start.tv_usec); //useconds
    run_time /= 1000000; // seconds
    printf("%lg\n", run_time);

  }
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


