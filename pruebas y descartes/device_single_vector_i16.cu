#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <cub/cub.cuh>
#include "common.h"

#define Elems 1000000000

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
  int run_time;

  // mempory reserve in CPU
  h_input = (int*) malloc(Elems*sizeof(int));
  h_output_sec = (int*) malloc(Elems*sizeof(int));
  h_output_par = (int*) malloc(Elems*sizeof(int));

  // memory reserve in GPU
  cudaMalloc(&d_input, Elems*sizeof(int));
  cudaMalloc(&d_output, Elems*sizeof(int));

  // initialization

  initialize_vector_i32(h_input, Elems);

  // Determine temporary device storage requirements
  void* d_temp_storage = NULL;
  size_t temp_storage_bytes = 0;
  
  cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_input, d_output, Elems);

  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);

  //////////
  // Parallel Part
  //////////

  printf("Paralelo\n");
  gettimeofday(&tv_start, NULL);
  
  //device copy
  cudaMemcpy(d_input, h_input, Elems*sizeof(int), cudaMemcpyHostToDevice);

  // Run sorting operation
  cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_input, d_output, Elems);   

  // recovery of the parallel results
  cudaMemcpy(h_output_par, d_output, Elems*sizeof(int), cudaMemcpyDeviceToHost);

  gettimeofday(&tv_end, NULL);
  run_time=(tv_end.tv_sec - tv_start.tv_sec) * 1000000 + (tv_end.tv_usec - tv_start.tv_usec); //useconds
  run_time /= 1000000; // seconds
  printf ( "Parallel Time: %lg s\n", run_time );
      
  //////////
  // Secuential Part
  //////////

  printf("Secuencial\n");
  gettimeofday(&tv_start, NULL);

  memcpy(h_output_sec, h_input, Elems*sizeof(int)); // copy of the original array
  qsort(h_output_sec, Elems, sizeof(int), compare_i32);

  gettimeofday(&tv_end, NULL);
  run_time=(tv_end.tv_sec - tv_start.tv_sec) * 1000000 + (tv_end.tv_usec - tv_start.tv_usec); //useconds
  run_time /= 1000000; // seconds
  printf ( "Secuential Time: %lg s\n", run_time );

  // return of the results
  int ret = compare_vector_i32(h_output_sec, h_output_par, Elems); 
  if(ret) {
    printf("Error on the execution: (diferent results after index: %d)\n", ret);
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
  cudaFree(d_temp_storage);

  return 0;
  
}


