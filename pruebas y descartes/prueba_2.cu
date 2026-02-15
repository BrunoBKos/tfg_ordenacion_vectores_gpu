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

// #define ARRS 200
#define ELEMS 27400

///////////
// Cabeceras
///////////

///////////
// funciones auxiliares
///////////

struct less_float {
    __device__ __host__ bool operator()(const float &a, const float &b) const {
        return a < b;
    }
};


///////////
// funcion principal
///////////

int main(int argc, char *argv[]) {

    struct timeval tv_start, tv_end;
    double run_time;
    
    float* vector;              // vector datos a ordenar

    int n_arrays;               // numero de subvectores

    if(argc < 2) {
        printf("Error en el paso de parámetros. Correcta ejecucion: ./programa numero_arrays\n");

    }

    n_arrays = atoi(argv[1]);
    printf("Numero de arrays: %d\n", n_arrays);

    printf("Inicio\n");
    gettimeofday(&tv_start, NULL);
    
    vector = (float*) malloc(n_arrays*ELEMS*sizeof(float));
    initialize_vector_f32(vector, n_arrays*ELEMS);

    gettimeofday(&tv_end, NULL);
    run_time=(tv_end.tv_sec - tv_start.tv_sec) * 1000000 + (tv_end.tv_usec - tv_start.tv_usec); //useconds
    run_time /= 1000000; // seconds
    printf("Tiempo: %lg segundos despues de la inicialización del vector\n", run_time);
    printf("-----------------------------------\n");

    /* EJECUCION SECUENCIAL CON THRUST EN CPU */
    printf("Ejecucion secuencial con thrust en CPU\n");
    {
        float* output = (float*) malloc(n_arrays*ELEMS*sizeof(float));
        gettimeofday(&tv_start, NULL); 
        memcpy(output, vector, n_arrays*ELEMS*sizeof(float)); // copy of the original array
        for(int j = 0; j < n_arrays; j++) {
            thrust::sort(thrust::host, output, output+ELEMS); // sorting
        }
        free(output);
    }
    gettimeofday(&tv_end, NULL);
    run_time=(tv_end.tv_sec - tv_start.tv_sec) * 1000000 + (tv_end.tv_usec - tv_start.tv_usec); //useconds
    run_time /= 1000000; // seconds
    printf("Tiempo: %lg segundos\n", run_time);
    printf("-----------------------------------\n");

    /* EJECUCION PARALELA CON OPENMP Y THRUST EN CPU */
    printf("Ejecucion paralela con openmp y thrust en CPU\n");
    gettimeofday(&tv_start, NULL);
    #pragma omp parallel
    {
        float* output = (float*) malloc(ELEMS*sizeof(float));
        // gettimeofday(&tv_start, NULL); 
        #pragma omp for
        for(int j = 0; j < n_arrays; j++) {
            memcpy(output, vector+(j*ELEMS), ELEMS*sizeof(float)); // copy of the original array
            thrust::sort(thrust::host, output, output+ELEMS); // sorting
        }
        free(output);
        #pragma omp barrier
    }
    gettimeofday(&tv_end, NULL);
    run_time=(tv_end.tv_sec - tv_start.tv_sec) * 1000000 + (tv_end.tv_usec - tv_start.tv_usec); //useconds
    run_time /= 1000000; // seconds
    printf("Tiempo: %lg segundos\n", run_time);
    printf("-----------------------------------\n");


    /* EJECUCION PARALELA CON CUB SEGMENTED RADIX SORT EN GPU */
    gettimeofday(&tv_start, NULL);
    printf("Ejecucion paralela con cub segmented radix sort en GPU\n");
    {
        float* output;
        float* d_input;
        float* d_output;
        int h_offsets[n_arrays+1];
        int* d_offsets;

        void* d_temp_storage = NULL;
        size_t temp_storage_bytes = 0;

        output = (float*) malloc(n_arrays*ELEMS*sizeof(float));

        cudaMalloc(&d_input, n_arrays*ELEMS*sizeof(float));
        cudaMalloc(&d_output, n_arrays*ELEMS*sizeof(float));
        cudaMalloc(&d_offsets, (n_arrays+1)*sizeof(int));

        for(int i = 0; i < (n_arrays+1); i++) h_offsets[i] = ELEMS*i;

        cub::DeviceSegmentedRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_input, d_output, ELEMS*n_arrays, n_arrays, 
                                                d_offsets, d_offsets+1, 0, sizeof(float)*8);

        // Allocate temporary storage
        cudaMalloc(&d_temp_storage, temp_storage_bytes);

        cudaMemcpy(d_offsets, h_offsets, (n_arrays+1)*sizeof(int), cudaMemcpyHostToDevice);
        //gettimeofday(&tv_start, NULL); 

        cudaMemcpy(d_input, vector, n_arrays*ELEMS*sizeof(float), cudaMemcpyHostToDevice);
        cub::DeviceSegmentedRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_input, d_output, ELEMS*n_arrays, n_arrays, 
                                                d_offsets, d_offsets+1, 0, sizeof(float)*8);
        cudaMemcpy(output, d_output, n_arrays*ELEMS*sizeof(float), cudaMemcpyDeviceToHost);
        
        free(output);
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_offsets);

    }
    gettimeofday(&tv_end, NULL);
    run_time=(tv_end.tv_sec - tv_start.tv_sec) * 1000000 + (tv_end.tv_usec - tv_start.tv_usec); //useconds
    run_time /= 1000000; // seconds
    printf("Tiempo: %lg segundos\n", run_time);
    printf("-----------------------------------\n");


    /* EJECUCION PARALELA CON MGPU SEGMENTED SORT EN GPU */
    gettimeofday(&tv_start, NULL);
    printf("Ejecucion paralela con mgpu segmented sort en GPU\n");
    {
        float* output;
        std::vector<int> h_offsets(n_arrays+1);
        float* d_input;
	    int* d_offsets;
        
        output = (float*) malloc(n_arrays*ELEMS*sizeof(float));
        cudaMalloc(&d_offsets, (n_arrays+1)*sizeof(int));
        cudaMalloc(&d_input, n_arrays*ELEMS*sizeof(float));

        for(int i = 0; i < (n_arrays+1); i++) h_offsets[i] = ELEMS*i;
        cudaMemcpy(d_offsets, h_offsets.data(), (n_arrays+1)*sizeof(int), cudaMemcpyHostToDevice);
        
        mgpu::standard_context_t context;

	// gettimeofday(&tv_start, NULL); 
        
        cudaMemcpy(d_input, vector, n_arrays*ELEMS*sizeof(float), cudaMemcpyHostToDevice);
        mgpu::segmented_sort(d_input, n_arrays*ELEMS, d_offsets, n_arrays, less_float(), context);
        cudaMemcpy(output, d_input, n_arrays*ELEMS*sizeof(float), cudaMemcpyDeviceToHost);
        
        free(output);
        cudaFree(d_input);
        cudaFree(d_offsets);

    }
    gettimeofday(&tv_end, NULL);
    run_time=(tv_end.tv_sec - tv_start.tv_sec) * 1000000 + (tv_end.tv_usec - tv_start.tv_usec); //useconds
    run_time /= 1000000; // seconds
    printf("Tiempo: %lg segundos\n", run_time);
    printf("-----------------------------------\n");

    /* EJECUCION PARALELA CON FAST SEGMENTED SORT EN GPU */
    gettimeofday(&tv_start, NULL);
    printf("Ejecucion paralela con fast segmented sort en GPU\n");
    {
        float* output;
        std::vector<int> h_offsets(n_arrays+1);
        int* d_offsets;
        float* d_input;
        float* d_values;

        output = (float*) malloc(n_arrays*ELEMS*sizeof(float));
        cudaMalloc(&d_offsets, (n_arrays+1)*sizeof(int));

        for(int i = 0; i < (n_arrays+1); i++) h_offsets[i] = ELEMS*i;
        cudaMemcpy(d_offsets, h_offsets.data(), (n_arrays+1)*sizeof(int), cudaMemcpyHostToDevice);
        
        cudaMalloc(&d_input, n_arrays*ELEMS*sizeof(float));
        cudaMalloc(&d_values, n_arrays*ELEMS*sizeof(float));

        // gettimeofday(&tv_start, NULL); 

        cudaMemcpy(d_input, vector, n_arrays*ELEMS*sizeof(float), cudaMemcpyHostToDevice);
        bb_segsort<float, float>(d_input, d_values, n_arrays*ELEMS, d_offsets, n_arrays);
        cudaMemcpy(output , d_input, n_arrays*ELEMS*sizeof(float), cudaMemcpyDeviceToHost);

        free(output);
        cudaFree(d_input);
        cudaFree(d_values);
        cudaFree(d_offsets);
    }
    gettimeofday(&tv_end, NULL);
    run_time=(tv_end.tv_sec - tv_start.tv_sec) * 1000000 + (tv_end.tv_usec - tv_start.tv_usec); //useconds
    run_time /= 1000000; // seconds
    printf("Tiempo: %lg segundos\n", run_time);
    printf("-----------------------------------\n");

    /* EJECUCION HIBRIDA CON CUB SEGMENTED READIX SORT EN GPU Y THRUST EN CPU */
    gettimeofday(&tv_start, NULL);
    printf("Ejecucion hibrida con cub segmented radix sort en GPU y thrust en cpu\n");
    {
        float* output;
        float* d_input;
        float* d_output;
        int h_offsets[n_arrays+1];
        int* d_offsets;

        void* d_temp_storage = NULL;
        size_t temp_storage_bytes = 0;
        cudaStream_t stream;

        output = (float*) malloc(n_arrays*ELEMS*sizeof(float));
        cudaMalloc(&d_input, (n_arrays/2)*ELEMS*sizeof(float));
        cudaMalloc(&d_output, (n_arrays/2)*ELEMS*sizeof(float));
        cudaMalloc(&d_offsets, ((n_arrays/2)+1)*sizeof(int));
        cudaStreamCreate(&stream);

        for(int i = 0; i < ((n_arrays/2)+1); i++) h_offsets[i] = ELEMS*i;
        cub::DeviceSegmentedRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_input, d_output, ELEMS*(n_arrays/2), (n_arrays/2), 
                                                d_offsets, d_offsets+1, 0, sizeof(float)*8);

        // Allocate temporary storage
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        cudaMemcpy(d_offsets, h_offsets, ((n_arrays/2)+1)*sizeof(int), cudaMemcpyHostToDevice);

        //gettimeofday(&tv_start, NULL); 

        cudaMemcpyAsync(d_input, vector, (n_arrays/2)*ELEMS*sizeof(float), cudaMemcpyHostToDevice, stream);
        cub::DeviceSegmentedRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_input, d_output, (n_arrays/2)*ELEMS, (n_arrays/2), 
                                                    d_offsets, d_offsets+1, 0, sizeof(float)*8, stream);
        cudaMemcpyAsync(output, d_output, (n_arrays/2)*ELEMS*sizeof(float), cudaMemcpyDeviceToHost, stream);

        memcpy(output+((n_arrays/2)*ELEMS), vector+((n_arrays/2)*ELEMS), (n_arrays/2)*ELEMS*sizeof(float)); // copy of the original array
        for(int j = (n_arrays/2); j < n_arrays; j++)
            thrust::sort(thrust::host, output+(j*ELEMS), output+((j+1)*ELEMS)); // sorting
        
        cudaStreamSynchronize(stream);

        free(output);
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_offsets);
        cudaStreamDestroy(stream);

    }
    gettimeofday(&tv_end, NULL);
    run_time=(tv_end.tv_sec - tv_start.tv_sec) * 1000000 + (tv_end.tv_usec - tv_start.tv_usec); //useconds
    run_time /= 1000000; // seconds
    printf("Tiempo: %lg segundos\n", run_time);
    printf("-----------------------------------\n");

    /* EJECUCION HIBRIDA CON CUB SEGMENTED READIX SORT EN GPU Y OPENMP CON THRUST EN CPU V2*/
    gettimeofday(&tv_start, NULL);
    printf("Ejecucion hibrida con cub segmented radix sort en GPU y openmp con thrust en cpu v2\n");
    {
        float* output;
        float* d_input;
        float* d_output;
        int h_offsets[n_arrays+1];
        int* d_offsets;

        output = (float*) malloc(n_arrays*ELEMS*sizeof(float));
        cudaMalloc(&d_input, (n_arrays/4)*ELEMS*sizeof(float));
        cudaMalloc(&d_offsets, ((n_arrays/4)+1)*sizeof(int));

        for(int i = 0; i < ((n_arrays/4)+1); i++) h_offsets[i] = ELEMS*i;

        cudaMemcpy(d_offsets, h_offsets, ((n_arrays/4)+1)*sizeof(int), cudaMemcpyHostToDevice);

        mgpu::standard_context_t context;

        cudaMemcpy(d_input, vector, (n_arrays/4)*ELEMS*sizeof(float), cudaMemcpyHostToDevice);
        mgpu::segmented_sort(d_input, (n_arrays/4)*ELEMS, d_offsets, n_arrays/2, less_float(), context);

        memcpy(output+((n_arrays/4)*ELEMS), vector+((n_arrays/2)*ELEMS), (n_arrays/4)*ELEMS*sizeof(float)); // copy of the original array
        #pragma omp parallel 
        {
        #pragma omp for
        for(int j = (n_arrays/4); j < n_arrays; j++) {
            thrust::sort(thrust::host, output+(j*ELEMS), output+((j+1)*ELEMS)); // sorting
        }
        }
        cudaMemcpy(output, d_input, (3*(n_arrays/4))*ELEMS*sizeof(float), cudaMemcpyDeviceToHost);

        free(output);
        cudaFree(d_input);
        cudaFree(d_offsets);

    }
    gettimeofday(&tv_end, NULL);
    run_time=(tv_end.tv_sec - tv_start.tv_sec) * 1000000 + (tv_end.tv_usec - tv_start.tv_usec); //useconds
    run_time /= 1000000; // seconds
    printf("Tiempo: %lg segundos\n", run_time);
    printf("-----------------------------------\n");


    /* EJECUCION PARALELA VERSION BACK TO BACK CON CUB DEVICE RADIX SORT PAIRS EN GPU */
    printf("Ejecucion paralela version back to back con device radix sort pairs en GPU\n");
    gettimeofday(&tv_start, NULL); 
    {
        float* output;
        float* d_input;
        float* d_output;
        int* h_keys;
        int* d_keys_in;
        int* d_keys_out;

        void* d_temp_storage_1 = NULL;
        void* d_temp_storage_2 = NULL;
        size_t temp_storage_bytes = 0;

        output = (float*) malloc(n_arrays*ELEMS*sizeof(float));
        h_keys = (int*) malloc(n_arrays*ELEMS*sizeof(int));
        cudaMalloc(&d_input, n_arrays*ELEMS*sizeof(float));
        cudaMalloc(&d_output, n_arrays*ELEMS*sizeof(float));
        cudaMalloc(&d_keys_in, n_arrays*ELEMS*sizeof(int));
        cudaMalloc(&d_keys_out, n_arrays*ELEMS*sizeof(int));

        for(int i = 0; i < n_arrays; i++) {
		    for(int j = 0; j < ELEMS; j++) h_keys[(i*ELEMS)+j] = i;
	    }

        cub::DeviceRadixSort::SortPairs(d_temp_storage_1, temp_storage_bytes, d_input, d_output, d_keys_in, d_keys_out, n_arrays*ELEMS);
        // Allocate temporary storage
        cudaMalloc(&d_temp_storage_1, temp_storage_bytes);

        cub::DeviceRadixSort::SortPairs(d_temp_storage_2, temp_storage_bytes, d_keys_in, d_keys_out, d_input, d_output, n_arrays*ELEMS);
        // Allocate temporary storage
        cudaMalloc(&d_temp_storage_2, temp_storage_bytes);

        cudaMemcpy(d_input, vector, n_arrays*ELEMS*sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_keys_in, h_keys, n_arrays*ELEMS*sizeof(int), cudaMemcpyHostToDevice);
        cub::DeviceRadixSort::SortPairs(d_temp_storage_1, temp_storage_bytes, d_input, d_output, d_keys_in, d_keys_out, n_arrays*ELEMS);
        cub::DeviceRadixSort::SortPairs(d_temp_storage_2, temp_storage_bytes, d_keys_out, d_keys_in, d_output, d_input, n_arrays*ELEMS);
        cudaMemcpy(output, d_input, n_arrays*ELEMS*sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_keys, d_keys_in, n_arrays*ELEMS*sizeof(int), cudaMemcpyDeviceToHost);

        free(output);
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_keys_in);
        cudaFree(d_keys_out);

    }
    gettimeofday(&tv_end, NULL);
    run_time=(tv_end.tv_sec - tv_start.tv_sec) * 1000000 + (tv_end.tv_usec - tv_start.tv_usec); //useconds
    run_time /= 1000000; // seconds
    printf("Tiempo: %lg segundos\n", run_time);
    printf("-----------------------------------\n");

    free(vector);
    
    return 0;
}
