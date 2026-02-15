#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <cub/cub.cuh>
#include "common.h"

#define Elems 1024
#define NTHREADS 512
#define ELEMSRED (Elems+NTHREADS-1)/NTHREADS
// valor máximo de un short
#define MAX 9999

// headers
__global__ void sort_par(short* input, short* output);

// main function
int main(void) {

  // host vectors
  short* h_input;
  short* h_output_sec;
  short* h_output_par;

  // device vectors
  short* d_input;
  short* d_output;

  // variables
  struct timeval tv_start, tv_end;
  short run_time;

  // mempory reserve in CPU
  h_input = (short*) malloc(Elems*sizeof(short));
  h_output_sec = (short*) malloc(Elems*sizeof(short));
  h_output_par = (short*) malloc(Elems*sizeof(short));

  // memory reserve in GPU
  cudaMalloc(&d_input, Elems*sizeof(short));
  cudaMalloc(&d_output, Elems*sizeof(short));

  // initialization

  initialize_vector_i16(h_input, Elems);

  //////////
  // Parallel Part
  //////////

  printf("Paralelo\n");
  gettimeofday(&tv_start, NULL);
  
  //device copy
  cudaMemcpy(d_input, h_input, Elems*sizeof(short), cudaMemcpyHostToDevice);

  // Run sorting operation
  // cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_input, d_output, Elems);   

  sort_par<<<1, NTHREADS>>>(d_input, d_output);

  // recovery of the parallel results
  cudaMemcpy(h_output_par, d_output, Elems*sizeof(short), cudaMemcpyDeviceToHost);

  gettimeofday(&tv_end, NULL);
  run_time=(tv_end.tv_sec - tv_start.tv_sec) * 1000000 + (tv_end.tv_usec - tv_start.tv_usec); //useconds
  run_time /= 1000000; // seconds
  printf ( "Parallel Time: %lg s\n", run_time );
      
  //////////
  // Secuential Part
  //////////

  printf("Secuencial\n");
  printf("valor de ELEMSRES:%d\n", ELEMSRED);
  gettimeofday(&tv_start, NULL);

  memcpy(h_output_sec, h_input, Elems*sizeof(short)); // copy of the original array
  qsort(h_output_sec, Elems, sizeof(short), compare_i16);

  gettimeofday(&tv_end, NULL);
  run_time=(tv_end.tv_sec - tv_start.tv_sec) * 1000000 + (tv_end.tv_usec - tv_start.tv_usec); //useconds
  run_time /= 1000000; // seconds
  printf ( "Secuential Time: %lg s\n", run_time );


  // return of the results
  int ret = compare_vector_i16(h_output_sec, h_output_par, Elems); 
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

  return 0;
  
}



__global__ void sort_par(short* input, short* output) {
    
    // sorting function
    typedef cub::BlockRadixSort<short, NTHREADS, ELEMSRED> BlockRadixSort;

    __shared__ typename BlockRadixSort::TempStorage temp_storage;
    
    short* local_input = input + blockIdx.x*Elems;
    short* local_output = output +  blockIdx.x*Elems;

    short thread_keys[ELEMSRED];

    for(int i = 0; i < ELEMSRED; i++) {
        thread_keys[i] = (((i+threadIdx.x*ELEMSRED) < Elems) ? local_input[i+threadIdx.x*ELEMSRED] : MAX);
    }

    __syncthreads();
    BlockRadixSort(temp_storage).Sort(thread_keys);
    __syncthreads();
    // storing of the resoults
    for(int i = 0; i < ELEMSRED; i++) {
        if((i+threadIdx.x*ELEMSRED) < Elems) local_output[i+threadIdx.x*ELEMSRED] = thread_keys[i];
    }


}
