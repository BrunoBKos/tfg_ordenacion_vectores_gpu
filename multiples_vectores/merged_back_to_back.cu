#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <math.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include "common.h"
#include <sys/time.h>
#include <cub/device/device_segmented_radix_sort.cuh>

///////////
// Declaración de Constantes
///////////

int N = 27000;
int M = 780;

__global__ void emptyKernel() {}

// #define M 780
#define N ((size_t) 27400)

int main(int argc, char** argv) {

    struct timeval tv_start, tv_end;
    double run_time;

    if(argc > 1) N = atoi(argv[1]);
    if(argc > 1) M = atoi(argv[2]);
   
    emptyKernel<<<1, 1>>>(); // GPU warmup

    short* vector;
    short* sorted_vector;

    // host reserves
    vector = (short*) malloc(M*N*sizeof(short));
    sorted_vector = (short*) malloc(M*N*sizeof(short));

    // initialization
    srand((unsigned int) time(NULL));
    for(int i = 0; i < M*N; i++) vect[i] = (short) (rand()%1024);
  
    cudaDeviceSynchronize();

    gettimeofday(&tv_start, NULL);
    /* EJECUCION PARALELA VERSION BACK TO BACK CON CUB DEVICE RADIX SORT PAIRS EN GPU */
    int* h_merged_vector;
    int* d_vector;
    int* d_sorted_vector;

    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    h_merged_vector = (int*) malloc(M*N*sizeof(int));
    cudaMalloc(&d_input, M*N*sizeof(int));
    cudaMalloc(&d_output, M*N*sizeof(int));

    for(int i = 0; i < M; i++) { // h_merged_vector = [31:16] (id_array) [15:0] (vector) 
	    for(int j = 0; j < N; j++) h_merged_vector[(i*N)+j] = ((i<<16)+(vector[(i*N)+j]));
	}
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_input, d_output, M*N);ç

    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    cudaMemcpy(d_vector, h_merged_vector, M*N*sizeof(int), cudaMemcpyHostToDevice);
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_vector, d_sorted_vector, M*N);
    cudaMemcpy(h_merged_vector, d_sorted_vector, M*N*sizeof(int), cudaMemcpyDeviceToHost);

    for(int i = 0; i < (M*N); i++) { // h_merged_vector = [31:16] (id_array) [15:0] (vector) 
	    sorted_vector[i] = (short) ((h_merged_vector[i]) & 65535);
	}

    free(h_vector_merged);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_temp_storage);

    gettimeofday(&tv_end, NULL);
    run_time=(tv_end.tv_sec - tv_start.tv_sec) * 1000000 + (tv_end.tv_usec - tv_start.tv_usec); //useconds
    run_time /= 1000000; // seconds
    printf("Tiempo: %lg segundos\n", run_time);
    printf("-----------------------------------\n");

    free(vector);
    free(sorted_vector);
    return 0;
}
