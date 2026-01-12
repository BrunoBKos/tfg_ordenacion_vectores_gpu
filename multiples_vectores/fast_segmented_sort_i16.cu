#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include "../../bb_segsort/bb_segsort.h"

int N = 27000;
int M = 780;

__global__ void emptyKernel() {}

// main function
int main(int argc, char** argv) {

  if(argc > 1) N = atoi(argv[1]);
  if(argc > 1) M = atoi(argv[2]);

  // host vectors
  short* vect;
  short* sorted_vector;

  // device vectors
  short* d_vector;
  short* d_values;
  int h_offsets[M+1];
  int* d_offsets;

  // time variables
  struct timeval tv_start, tv_end;
  double run_time;

  // memmory reserve in CPU
  vect = (short*) malloc(M*N*sizeof(short));
  sorted_vector = (short*) malloc(M*N*sizeof(short));

  // memory reserve in GPU
  cudaMalloc(&d_vector, M*N*sizeof(short));
  cudaMalloc(&d_values, M*N*sizeof(short));
  cudaMalloc(&d_offsets, (M+1)*sizeof(int));

  emptyKernel<<<1, 1>>>(); // GPU warmup

  for(int i = 0; i < (M+1); i++) h_offsets[i] = N*i;
  cudaMemcpy(d_offsets, h_offsets, (M+1)*sizeof(int), cudaMemcpyHostToDevice);

  // initialization
  srand((unsigned int) time(NULL));
  for(int i = 0; i < M*N; i++) vect[i]= (short) (rand()%1024);
  
  cudaDeviceSynchronize();
  
  gettimeofday(&tv_start, NULL);

  //device copy
  cudaMemcpy(d_vector, vect, M*N*sizeof(short), cudaMemcpyHostToDevice);
  // Run sorting operation
  bb_segsort<short, short>(d_vector, d_values, M*N, d_offsets, M);
  // recovery of the parallel results
  cudaMemcpy(sorted_vector, d_vector, M*N*sizeof(short), cudaMemcpyDeviceToHost);

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


