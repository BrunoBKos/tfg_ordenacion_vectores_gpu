#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <cub/cub.cuh>

#define Arrs 1
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
  int* d_input;
  int* d_output;

  // variables
  // int num_blks = 1;

  // mempory reserve in CPU
  h_input = (int*) malloc(Arrs*Elems*sizeof(int));
  h_output_sec = (int*) malloc(Arrs*Elems*sizeof(int));
  h_output_par = (int*) malloc(Arrs*Elems*sizeof(int));

  // memory reserve in GPU
  cudaMalloc(&d_input, Elems*sizeof(int));
  cudaMalloc(&d_output, Elems*sizeof(int));

  // initialization
  initialize(h_input, Arrs*Elems);
 
  //device copy
  cudaMemcpy(d_input, h_input, Arrs*Elems*sizeof(int), cudaMemcpyHostToDevice);

  for(int i = 0;  i < Arrs; i++) {
    // Determine temporary device storage requirements
    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_input+(i*Elems), d_output+(i*Elems), Elems);
    
    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    
    // Run sorting operation
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_input+(i*Elems), d_output+(i*Elems), Elems);
  
    cudaFree(d_temp_storage);

    cudaDeviceSynchronize();

  }

  // recovery of the parallel results
  cudaMemcpy(h_output_par, d_output, Arrs*Elems*sizeof(int), cudaMemcpyDeviceToHost);

  // secuential calculation of the results
  sort_sec(h_input, h_output_sec, Arrs, Elems);

  // return of the results
  int ret = compare(h_output_sec, h_output_par, Arrs*Elems); 
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
  cudaFree(d_input);
  cudaFree(d_output);

  return 0;
  
}

// CPU function for the secuential ordenation of n_arrays vectors of n_elems elements 
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
      int a = (rand() % 100);
      vect[i] = a;
    }
}

void print_all(int* h_output_sec, int* h_output_par) {
  for(int j = 0; j < (Arrs*Elems); j+=8) {
    printf("CPU calculated:   ");
    for(int i = 0; i < 6; i++) printf(" %d", h_output_sec[i+j]);
    printf(" \n"); 
    printf("GPU calculated:   ");
    for(int i = 0; i < 6; i++) printf(" %d", h_output_par[i+j]);
    printf(" \n");
  }
}