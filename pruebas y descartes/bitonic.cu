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

#include <vector>
#include <iostream>

#include <cuda_runtime.h>
#include <float.h>

#define ELEMS ((size_t) 27400)
#define NUM_VECTORS 1000
#define THREADS_PER_BLOCK 256

// Bitonic sort kernel - performs one comparison step
__global__ void bitonic_sort_step(float *vectors, int j, int k, int vector_size) {
    // Calculate global thread index
    unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
    
    // Calculate which vector this thread belongs to
    int vector_id = tid / vector_size;
    
    // Calculate position within the vector
    int i = tid % vector_size;
    
    // Check if this thread is within bounds
    if (vector_id >= NUM_VECTORS || i >= vector_size) {
        return;
    }
    
    // Pointer to the start of this vector
    float *vector = vectors + vector_id * vector_size;
    
    // Calculate the partner index using XOR
    unsigned int ixj = i ^ j;
    
    // Only threads with lower index than their partner do the comparison
    if (ixj > i && ixj < vector_size) {
        // Determine sort direction based on position in bitonic sequence
        if ((i & k) == 0) {
            // Sort ascending
            if (vector[i] > vector[ixj]) {
                // Swap elements
                float temp = vector[i];
                vector[i] = vector[ixj];
                vector[ixj] = temp;
            }
        } else {
            // Sort descending (to create bitonic sequence)
            if (vector[i] < vector[ixj]) {
                // Swap elements
                float temp = vector[i];
                vector[i] = vector[ixj];
                vector[ixj] = temp;
            }
        }
    }
}

// Find next power of 2
int next_power_of_2(int n) {
    int power = 1;
    while (power < n) {
        power *= 2;
    }
    return power;
}

// Perform bitonic sort on all vectors
void bitonic_sort(float *d_vectors, int num_vectors, int vector_size) {
    // Calculate total number of threads needed
    int total_elements = num_vectors * vector_size;
    
    // Calculate grid and block dimensions
    int threads_per_block = THREADS_PER_BLOCK;
    int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
    
    // Major steps - build bitonic sequences of increasing size
    for (int k = 2; k <= vector_size; k <<= 1) {
        // Minor steps - merge bitonic sequences
        for (int j = k >> 1; j > 0; j >>= 1) {
            bitonic_sort_step<<<num_blocks, threads_per_block>>>(
                d_vectors, j, k, vector_size);
            }
    }
    
    // Synchronize to ensure all kernels have completed
    cudaDeviceSynchronize();
}


__global__ void emptyKernel() {}


int main(int argc, char** argv) {

    struct timeval tv_start, tv_end;
    double run_time;

    float* vector;
    float* sorted_vector_sec;
    float* sorted_vector_par;

    int n_arrays = NUM_VECTORS;

    if(argc > 1) {
        n_arrays = atoi(argv[1]);
        printf("Numero de arrays: %d\n", n_arrays);
    }
    // host reserves
    vector = (float*) malloc(n_arrays*ELEMS*sizeof(float));
    sorted_vector_sec = (float*) malloc(n_arrays*ELEMS*sizeof(float));
    sorted_vector_par = (float*) malloc(n_arrays*ELEMS*sizeof(float));

    emptyKernel<<<1, 1>>>();

    // Vector Initizalize
    printf("Vector Initialize\n");
    gettimeofday(&tv_start, NULL);
    initialize_vector_f32(vector, n_arrays*ELEMS);
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
        float* output;
        #pragma omp for
        for(int j = 0; j < n_arrays; j++) {
            output = sorted_vector_sec + (j*ELEMS);
            memcpy(output, vector+(j*ELEMS), ELEMS*sizeof(float)); // copy of the original array
            thrust::sort(thrust::host, output, output+ELEMS); // sorting
        }
        #pragma omp barrier
    }
    gettimeofday(&tv_end, NULL);
    run_time=(tv_end.tv_sec - tv_start.tv_sec) * 1000000 + (tv_end.tv_usec - tv_start.tv_usec); //useconds
    run_time /= 1000000; // seconds
    printf("TIme in seconds after the openmp sorting execution: %lg \n", run_time);
    printf("-----------------------------------\n");

    // EJECUCION PARALELA BITIONIC SORT GPU //
    printf("Ejecucion paralela bitonic sort en GPU\n");
    gettimeofday(&tv_start, NULL); 
    {
        int padded_sz = next_power_of_2(ELEMS);
        int diff = (padded_sz - ELEMS);

        float* h_vectors;
        float* d_vectors;
        float rest[diff];
    
        h_vectors = (float*) malloc(n_arrays*padded_sz*sizeof(float));
        cudaMalloc(&d_vectors, n_arrays*padded_sz*sizeof(float));

	    #pragma omp parallel for
        for(int i = 0; i < diff; i++) rest[i] = FLT_MAX;

        #pragma omp parallel for
        for(int i = 0; i < n_arrays; i++) {
            memcpy(h_vectors + (i*padded_sz), vector + (i*ELEMS), ELEMS*sizeof(float));
            memcpy(h_vectors + ((i*padded_sz) + ELEMS), rest, diff*sizeof(float));
        }
	    cudaMemcpy(d_vectors, h_vectors, n_arrays*padded_sz*sizeof(float), cudaMemcpyHostToDevice);
        
        // bitonic sort batch
        bitonic_sort(d_vectors, n_arrays, padded_sz);
        
        cudaMemcpy(h_vectors, d_vectors, n_arrays*padded_sz*sizeof(float), cudaMemcpyDeviceToHost);

        #pragma omp parallel for
        for(int i = 0; i < n_arrays; i++) {
		    memcpy(sorted_vector_par+(i*ELEMS), h_vectors+(i*padded_sz), ELEMS*sizeof(float));
	    }

        free(h_vectors);
        cudaFree(d_vectors);
    }
    gettimeofday(&tv_end, NULL);
    run_time=(tv_end.tv_sec - tv_start.tv_sec) * 1000000 + (tv_end.tv_usec - tv_start.tv_usec); //useconds
    run_time /= 1000000; // seconds
    printf("Tiempo: %lg segundos\n", run_time);
    printf("-----------------------------------\n");

    int ret = memcmp(sorted_vector_sec, sorted_vector_par, n_arrays*ELEMS*sizeof(float));
    if(ret) {
            ret = abs(ret);
            printf("Error. Given value: %d\n", ret);
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