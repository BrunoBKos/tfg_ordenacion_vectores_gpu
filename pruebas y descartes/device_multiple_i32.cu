#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <cub/cub.cuh>
#include "common.h"
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <iostream>

#define Elems 27000
#define ARRS 780

// main function
int main(void) {

  // host vectors
  int* h_input;
  int* h_output_sec;
  int* h_output_par;

  // device vectors
  int* d_input;
  int* d_output;

  // variables
  struct timeval tv_start, tv_end;
  double run_time;
  double omp_start, omp_end;

  // mempory reserve in CPU
  h_input = (int*) malloc(ARRS*Elems*sizeof(int));
  h_output_sec = (int*) malloc(ARRS*Elems*sizeof(int));
  h_output_par = (int*) malloc(ARRS*Elems*sizeof(int));

  // initialization

  initialize_vector_i32(h_input, ARRS*Elems);

  // memory reserve in GPU
  cudaMalloc(&d_input, ARRS*Elems*sizeof(int));
  cudaMalloc(&d_output, ARRS*Elems*sizeof(int));

#pragma omp parallel
{


  // Determine temporary device storage requirements
  void* d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  cudaStream_t stream;

  cudaStreamCreate(&stream);

  cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_input, d_output, Elems, 0, sizeof(int)*8, stream);

  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);

  //////////
  // Parallel Part
  //////////
  #pragma omp master 
  {
  printf("Paralelo\n");
  //gettimeofday(&tv_start, NULL);
  omp_start = omp_get_wtime();
  //device copy
  cudaMemcpy(d_input, h_input, ARRS*Elems*sizeof(int), cudaMemcpyHostToDevice);
  }
  #pragma omp barrier
  #pragma omp for
  for(int i = 0; i < ARRS; i++) {

    // Run sorting operation
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_input+(i*Elems), d_output+(i*Elems), Elems, 0, sizeof(int)*8, stream);
  
    // Synchronize stream
    cudaStreamSynchronize(stream);
  }
  #pragma omp barrier
  #pragma omp master 
  {
  // recovery of the parallel results
  cudaMemcpy(h_output_par, d_output, ARRS*Elems*sizeof(int), cudaMemcpyDeviceToHost);
  omp_end = omp_get_wtime();
  printf("Parallel Time: %f s\n", omp_end - omp_start);
  }
  cudaFree(d_temp_storage);
  cudaStreamDestroy(stream);
  /*
  gettimeofday(&tv_end, NULL);
  run_time=(tv_end.tv_sec - tv_start.tv_sec) * 1000000 + (tv_end.tv_usec - tv_start.tv_usec); //useconds
  run_time /= 1000000; // seconds
  printf ( "Parallel Time: %lg s\n", run_time );
  */
  //////////
  // Secuential Part
  //////////
  #pragma omp barrier
  #pragma omp master 
  {
  printf("Secuencial\n");
  omp_start = omp_get_wtime();
  }
  //gettimeofday(&tv_start, NULL);
  #pragma omp for
  for(int i = 0; i < ARRS; i++) {
    memcpy(h_output_sec+(i*Elems), h_input+(i*Elems), Elems*sizeof(int)); // copy of the original array
    thrust::sort(thrust::host, h_output_sec+(i*Elems), h_output_sec+(i*Elems)+Elems);
    // qsort(h_output_sec+(i*Elems), Elems, sizeof(int), compare_i32);
  }
  #pragma omp barrier
  #pragma omp master 
  {
  omp_end = omp_get_wtime();
  printf("\"Secuential\" Time: %f s\n", omp_end - omp_start);
  }
  /*
  gettimeofday(&tv_end, NULL);
  run_time=(tv_end.tv_sec - tv_start.tv_sec) * 1000000 + (tv_end.tv_usec - tv_start.tv_usec); //useconds
  run_time /= 1000000; // seconds
  printf ( "Secuential Time: %lg s\n", run_time );
  */
  // return of the results
}
  int ret = compare_vector_i32(h_output_sec, h_output_par, ARRS*Elems);
  if(ret) {
    printf("Error on the execution: (diferent results in Array: %d after index: %d)\n",ret/ARRS, ret%ARRS);
    printf("Expected:");
    for(int i = 0; i < 6; i++) printf(" %d", h_output_sec[i+ret]);
    printf("\n");
    printf("Given:   ");
    for(int i = 0; i < 6; i++) printf(" %d", h_output_par[i+ret]);
    printf("\n");
    exit(1);
  }
  else{
    printf("Input:            ");
    for(int i = 0; i < 8; i++) printf(" %d", h_input[i+(Elems/2)]);
    printf(" ...\n"); 
    printf("CPU calculated:   ");
    for(int i = 0; i < 8; i++) printf(" %d", h_output_sec[i+(Elems/2)]);
    printf(" ...\n"); 
    printf("GPU calculated:   ");
    for(int i = 0; i < 8; i++) printf(" %d", h_output_par[i+(Elems/2)]);
    printf(" ...\n");
    printf("Correct execution: Success\n");
  } 

  // free host memory
  free(h_input);
  free(h_output_sec);
  free(h_output_par);

  // free device memory
  cudaFree(d_input);
  cudaFree(d_output);


  return 0;
  
}


