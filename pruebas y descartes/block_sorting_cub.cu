#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <cub/cub.cuh>

#define Arrs 1024
#define Elems 2048
#define NTHREADS 1024

// headers
void sort_sec(int* input, int* output, size_t n_arrays, size_t n_elems);

__global__ void sort_par(int* vector);

int compare(int* a, int* b, size_t n);

void initialize(int* a, size_t n);

// main function
int main(void) {

  // host vectors
  int* h_input;
  int* h_output_sec;
  int* h_output_par;

  // device vectors
  int* d_vector;

  // variables
  int num_blks;

  // mempory reserve in CPU
  h_input = (int*) malloc(Arrs*Elems*sizeof(int));
  h_output_sec = (int*) malloc(Arrs*Elems*sizeof(int));
  h_output_par = (int*) malloc(Arrs*Elems*sizeof(int));

  // memory reserve in GPU
  cudaMalloc(&d_vector, Arrs*Elems*sizeof(int));

  // initialization
  initialize(d_vector, Arrs*Elems);

  //device copy
  cudaMemcpy(d_vector, h_input, Arrs*Elems*sizeof(int), cudaMemcpyHostToDevice);

  // call to device kernel
  num_blks = Arrs/NTHREADS;
  sort_par<<<num_blks, NTHREADS>>>(d_vector);
  
  // secuential calculation of the results
  sort_sec(h_input, h_output_sec, Arrs, Elems);

  // recovery of the parallel results
  cudaMemcpy(h_output_par, d_vector, Arrs*Elems*sizeof(int), cudaMemcpyDeviceToHost);

  // return of the results
  int ret = compare(h_output_sec, h_output_par, Arrs*Elems); 
  if(ret) {
    printf("Error on the execution: (diferent results after index: %d)\n", ret);
    printf("Expected:");
    for(int i = 0; i < 6; i++) printf(" %f", h_output_sec[i+ret]);
    printf("\n");
    printf("Given:   ");
    for(int i = 0; i < 6; i++) printf(" %f", h_output_par[i+ret]);
    printf("\n");
    exit(1);
  }
  else{
    printf("Input:   ");
    for(int i = 0; i < 6; i++) printf(" %d", h_input[i+(Arrs*(Elems/2))]);
    printf(" ...\n"); 
    printf("CPU calculated:   ");
    for(int i = 0; i < 6; i++) printf(" %d", h_output_sec[i+(Arrs*(Elems/2))]);
    printf(" ...\n"); 
    printf("GPU calculated:   ");
    for(int i = 0; i < 6; i++) printf(" %d", h_output_par[i+(Arrs*(Elems/2))]);
    printf(" ...\n");
    printf("Correct execution: Success\n");
  } 

  // free host memory
  free(h_input);
  free(h_output_sec);
  free(h_output_par);

  // free device memory
  cudaFree(d_vector);

  return 0;
  
}

// CPU function for the secuential product between matrix and vector
void sort_sec(int* input, int* output, size_t n_arrays, size_t n_elems) {
  
  int i, j, k, aux;
  int* output_aux = output;

  memcpy(output, input, n_arrays*n_elems*sizeof(int)); // copy of the original array

  for(i = 0; i < n_arrays; i++) {
    for(j = 0; j < n_elems; j++) {
      for(k = (j+1); k < n_elems; k++) {
        // sorting function
        if(output_aux[j] > output_aux[k]) { // elements change position
            aux = output_aux[j]; 
            output_aux[j] = output_aux[k];
            output_aux[k] = aux;
        }
      }
    }
    output_aux += n_elems;
  }

}


__global__ void sort_par(int* vector) {

  int th_id = threadIdx.x + blockDim.x*blockIdx.x; 
  // sorting function
  typedef cub::BlockRadixSort<int, NTHREADS, Elems> BlockRadixSort;

  int thread_keys[Elems];
  for(int i = 0; i < Elems; i++) thread_keys[i] = vector[i+th_id*Elems];

  __shared__ typename BlockRadixSort::TempStorage temp_storage;

  BlockRadixSort(temp_storage).Sort(thread_keys);

  for(int i = 0; i < Elems; i++) vector[i+th_id*Elems] = thread_keys[i];


}



// auxiliar function to compare the values of two diferent vectors
int compare(int* a, int* b, size_t n) {
  int i;
  for(i = 0; i < n; i++)
    if(a[i] != b[i]) break;
  return (i - n) ? i : 0;
}

// auxiliar function to initialize vectors with random values (int)
void initialize(int* vect, size_t n){
    int i;
    for (i = 0; i < n; i++) {
	    vect[i] = (rand() % 100);
    }
}

