#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <cub/cub.cuh>

#define Arrs 800
#define Elems 27400
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

  int i_print=1;
  struct timeval tv_start, tv_end;
  double run_time;

  // variables
  // int num_blks = 1;

  printf("Paralelo\n");
  gettimeofday(&tv_start, NULL);

  // mempory reserve in CPU
  h_input = (int*) malloc(Arrs*Elems*sizeof(int));
  h_output_sec = (int*) malloc(Arrs*Elems*sizeof(int));
  h_output_par = (int*) malloc(Arrs*Elems*sizeof(int));

  // memory reserve in GPU
  cudaMalloc(&d_input, Arrs*Elems*sizeof(int));
  cudaMalloc(&d_output, Arrs*Elems*sizeof(int));

  // initialization
  initialize(h_input, Arrs*Elems);
 
  //device copy
  cudaMemcpy(d_input, h_input, Arrs*Elems*sizeof(int), cudaMemcpyHostToDevice);

  gettimeofday(&tv_end, NULL);
  run_time=(tv_end.tv_sec - tv_start.tv_sec) * 1000000 +
    (tv_end.tv_usec - tv_start.tv_usec); //useconds
  run_time /= 1000000; // seconds
  printf ( " Tras CudaMemCpy t: %lg s\n", run_time );

  for(int i = 0;  i < Arrs; i++) {
   
    // Determine temporary device storage requirements
    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_input+(i*Elems), d_output+(i*Elems), Elems);

    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // Run sorting operation
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_input+(i*Elems), d_output+(i*Elems), Elems);
      
    cudaDeviceSynchronize();
      
    cudaFree(d_temp_storage);

    if ( i == i_print ) {
      gettimeofday(&tv_end, NULL);
      run_time=(tv_end.tv_sec - tv_start.tv_sec) * 1000000 +
        (tv_end.tv_usec - tv_start.tv_usec); //useconds
      run_time /= 1000000; // seconds
      printf ( " i: %8d t: %lg s\n", i, run_time );
      i_print = 2 * i_print;
    }

  }
  // recovery of the parallel results
  cudaMemcpy(h_output_par, d_output, Arrs*Elems*sizeof(int), cudaMemcpyDeviceToHost);

  gettimeofday(&tv_end, NULL);
  run_time=(tv_end.tv_sec - tv_start.tv_sec) * 1000000 +
    (tv_end.tv_usec - tv_start.tv_usec); //useconds
  run_time /= 1000000; // seconds
  printf ( " i: %8d t: %lg s\n", Arrs, run_time );
      
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
    for(int i = 0; i < 20; i++) printf(" %d", h_input[i+(Arrs*(Elems/2))]);
    /* for(int i = 0; i < Arrs; i++) */
    /*   for(int j = 0; j < Elems; j++) */
    /* printf(" %d", h_input[i*Elems+j]); */
    printf(" ...\n"); 
    printf("CPU calculated:   ");
    for(int i = 0; i < 20; i++) printf(" %d", h_output_sec[i+(Arrs*(Elems/2))]);
    /* for(int i = 0; i < Arrs; i++) */
    /*   for(int j = 0; j < Elems; j++) */
    /* printf(" %d", h_output_sec[i*Elems+j]); */
    printf(" ...\n"); 
    printf("GPU calculated:   ");
    for(int i = 0; i < 20; i++) printf(" %d", h_output_par[i+(Arrs*(Elems/2))]);
    /* for(int i = 0; i < Arrs; i++) */
    /*   for(int j = 0; j < Elems; j++) */
    /* printf(" %d", h_output_par[i*Elems+j]); */
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
  int i_print=1;
  
  struct timeval tv_start, tv_end;
  double run_time;

  printf("Secuencial\n");
  gettimeofday(&tv_start, NULL);
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
    if ( i == i_print ) {
      gettimeofday(&tv_end, NULL);
      run_time=(tv_end.tv_sec - tv_start.tv_sec) * 1000000 +
        (tv_end.tv_usec - tv_start.tv_usec); //useconds
      run_time /= 1000000; // seconds
      printf ( " i: %8d t: %lg s\n", i, run_time );
      i_print = 2 * i_print;
    }
  }
  gettimeofday(&tv_end, NULL);
  run_time=(tv_end.tv_sec - tv_start.tv_sec) * 1000000 +
    (tv_end.tv_usec - tv_start.tv_usec); //useconds
  run_time /= 1000000; // seconds
  printf ( " i: %8d t: %lg s\n", n_arrays, run_time );

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
