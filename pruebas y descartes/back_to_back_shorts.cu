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

#include <moderngpu/kernel_segsort.hxx>
#include <moderngpu/context.hxx>
#include <vector>
#include <iostream>

#include "../bb_segsort/bb_segsort.h"

///////////
// Declaración de Constantes
///////////

using namespace mgpu;

// #define n_arrays 780
#define ELEMS ((size_t) 27400)



int main(int argc, char** argv) {

    struct timeval tv_start, tv_end;
    double run_time;

    short* vector;
    short* sorted_vector_sec;
    short* sorted_vector_par;

    int n_arrays;

    if(argc < 3) {
        printf("Error en el paso de parámetros. Correcta ejecucion: ./programa numero_arrays version\n");
    }
    n_arrays = atoi(argv[1]);
    printf("Numero de arrays: %d\n", n_arrays);
   
    // host reserves
    vector = (short*) malloc(n_arrays*ELEMS*sizeof(short));
    sorted_vector_sec = (short*) malloc(n_arrays*ELEMS*sizeof(short));
    sorted_vector_par = (short*) malloc(n_arrays*ELEMS*sizeof(short));

    // Vector Initizalize
    printf("Vector Initialize\n");
    gettimeofday(&tv_start, NULL);
    initialize_vector_i16(vector, n_arrays*ELEMS);
    gettimeofday(&tv_end, NULL);
    run_time=(tv_end.tv_sec - tv_start.tv_sec) * 1000000 + (tv_end.tv_usec - tv_start.tv_usec); //useconds
    run_time /= 1000000; // seconds
    printf("TIme in seconds after the initialization of the input vectors: %lg \n", run_time);
    printf("-----------------------------------\n");

    // OpenMP EXECUTION
    printf("Ejecucion paralela con openmp y thrust en CPU\n");
    gettimeofday(&tv_start, NULL);
    #pragma omp parallel 
    {
        short* output;
        #pragma omp for
        for(int j = 0; j < n_arrays; j++) {
            output = sorted_vector_sec + (j*ELEMS);
            memcpy(output, vector+(j*ELEMS), ELEMS*sizeof(short)); // copy of the original array
            thrust::sort(thrust::host, output, output+ELEMS); // sorting
        }
        #pragma omp barrier
    }
    gettimeofday(&tv_end, NULL);
    run_time=(tv_end.tv_sec - tv_start.tv_sec) * 1000000 + (tv_end.tv_usec - tv_start.tv_usec); //useconds
    run_time /= 1000000; // seconds
    printf("TIme in seconds after the openmp sorting execution: %lg \n", run_time);
    printf("-----------------------------------\n");

    /* EJECUCION PARALELA VERSION BACK TO BACK CON CUB DEVICE RADIX SORT PAIRS EN GPU */
    printf("Ejecucion paralela version back to back con device radix sort pairs en GPU\n");
    gettimeofday(&tv_start, NULL); 
    {
        int* h_vector;
        int* d_input;
        int* d_output;

        void* d_temp_storage = NULL;
        size_t temp_storage_bytes = 0;

        h_vector = (int*) malloc(n_arrays*ELEMS*sizeof(int));
        cudaMalloc(&d_input, n_arrays*ELEMS*sizeof(int));
        cudaMalloc(&d_output, n_arrays*ELEMS*sizeof(int));
        
        #pragma omp parallel for
        for(int i = 0; i < n_arrays; i++) { // h_vector = [31:16] (id_array) [15:0] (vector) 
		    for(int j = 0; j < ELEMS; j++) h_vector[(i*ELEMS)+j] = ((i<<16)+(vector[(i*ELEMS)+j]));
	    }

        cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_input, d_output, n_arrays*ELEMS);
        // Allocate temporary storage
        cudaMalloc(&d_temp_storage, temp_storage_bytes);

        cudaMemcpy(d_input, h_vector, n_arrays*ELEMS*sizeof(int), cudaMemcpyHostToDevice);
        cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_input, d_output, n_arrays*ELEMS);
        cudaMemcpy(h_vector, d_output, n_arrays*ELEMS*sizeof(int), cudaMemcpyDeviceToHost);

        #pragma omp parallel for
        for(int i = 0; i < (n_arrays*ELEMS); i++) { // h_vector = [31:16] (id_array) [15:0] (vector) 
		    sorted_vector_par[i] = (short) ((h_vector[i]) & 65535);
	    }

        free(h_vector);
        cudaFree(d_input);
        cudaFree(d_output);

    }
    gettimeofday(&tv_end, NULL);
    run_time=(tv_end.tv_sec - tv_start.tv_sec) * 1000000 + (tv_end.tv_usec - tv_start.tv_usec); //useconds
    run_time /= 1000000; // seconds
    printf("Tiempo: %lg segundos\n", run_time);
    printf("-----------------------------------\n");

    int ret = memcmp(sorted_vector_sec, sorted_vector_par, n_arrays*ELEMS*sizeof(short));
    if(ret) {
       printf("Error. Given value: %d\n", ret/sizeof(short));
	printf("Expected: ");
      	for(int i = ret; i < ret+10; i++) {
		    printf("%f ", sorted_vector_sec[i]);
	}
	printf("Given: ");
	    for(int i = ret; i < ret+10; i++) {
		    printf("%f ", sorted_vector_par[i]);
	}
    }
    else printf("Correcto\n");

    free(vector);
    free(sorted_vector_sec);
    free(sorted_vector_par);
    return 0;
}
