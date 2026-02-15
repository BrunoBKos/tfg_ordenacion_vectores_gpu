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

#define EJEX 150
#define EJEY 200
#define ELEMS 27400

///////////
// funciones auxiliares
///////////

///////////
// funcion principal
///////////

int main(int argc, char *argv[]) {

    struct timeval tv_start, tv_end;
    double run_time;
    
    float* vector;

    printf("Inicio\n");
    gettimeofday(&tv_start, NULL);
    vector = (float*) malloc(EJEX*EJEY*ELEMS*sizeof(float));
    initialize_vector_f32(vector, EJEX*EJEY*ELEMS);
    gettimeofday(&tv_end, NULL);
    run_time=(tv_end.tv_sec - tv_start.tv_sec) * 1000000 + (tv_end.tv_usec - tv_start.tv_usec); //useconds
    run_time /= 1000000; // seconds
    printf("Tiempo: %lg segundos despues de la inicialización del vector\n", run_time);
    printf("-----------------------------------\n");

    /* EJECUCION PARALELA CON OPENMP Y THRUST EN CPU */
    printf("Ejecucion paralela con openmp y thrust en CPU\n");
    gettimeofday(&tv_start, NULL);
#pragma omp parallel 
{
    float* output = (float*) malloc(ELEMS*sizeof(float));
    #pragma omp for
    for(int i = 0; i < EJEX; i++) {
        for(int j = 0; j < EJEY; j++) {
            memcpy(output, vector+((i*EJEY + j)*ELEMS), ELEMS*sizeof(float)); // copy of the original array
            thrust::sort(thrust::host, output, output+ELEMS); // sorting
        }
    }
    free(output);
    #pragma omp barrier
}
    gettimeofday(&tv_end, NULL);
    run_time=(tv_end.tv_sec - tv_start.tv_sec) * 1000000 + (tv_end.tv_usec - tv_start.tv_usec); //useconds
    run_time /= 1000000; // seconds
    printf("Tiempo: %lg segundos\n", run_time);
    printf("-----------------------------------\n");

    /* EJECUCION EN GPU CON LA FUNCION DEVICE RADIX SORT SIN STREAMS */
    printf("Ejecucion CUB device radix sort sin streams\n");
    gettimeofday(&tv_start, NULL);
#pragma omp parallel 
{
    float* output;
    float* d_input;
    float* d_output;

    // Determine temporary device storage requirements
    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    output = (float*) malloc(ELEMS*sizeof(float));
    cudaMalloc(&d_input, ELEMS*sizeof(float));
    cudaMalloc(&d_output, ELEMS*sizeof(float));

    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_input, d_output, ELEMS, 0, sizeof(float)*8);

    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    
    #pragma omp for
    for(int i = 0; i < EJEX; i++) {
        for(int j = 0; j < EJEY; j++) {
            cudaMemcpy(d_input, vector+((i*EJEY + j)*ELEMS), ELEMS*sizeof(float), cudaMemcpyHostToDevice);
            cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_input, d_output, ELEMS, 0, sizeof(float)*8);
            cudaMemcpy(output, d_output, ELEMS*sizeof(float), cudaMemcpyDeviceToHost);
        }
    }
    free(output);
    cudaFree(d_input);
    cudaFree(d_output);
    #pragma omp barrier
}
    gettimeofday(&tv_end, NULL);
    run_time=(tv_end.tv_sec - tv_start.tv_sec) * 1000000 + (tv_end.tv_usec - tv_start.tv_usec); //useconds
    run_time /= 1000000; // seconds
    printf("Tiempo: %lg segundos\n", run_time);
    printf("-----------------------------------\n");

    /* EJECUCION EN GPU CON LA FUNCION DEVICE RADIX SORT Y STREAMS */
    printf("Ejecucion CUB device radix sort\n");
    gettimeofday(&tv_start, NULL);
#pragma omp parallel 
{
    float* output;
    float* d_input;
    float* d_output;

    // Determine temporary device storage requirements
    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cudaStream_t stream;

    output = (float*) malloc(ELEMS*sizeof(float));
    cudaMalloc(&d_input, ELEMS*sizeof(float));
    cudaMalloc(&d_output, ELEMS*sizeof(float));
    
    cudaStreamCreate(&stream);

    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_input, d_output, ELEMS, 0, sizeof(float)*8, stream);

    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    
    #pragma omp for
    for(int i = 0; i < EJEX; i++) {
        for(int j = 0; j < EJEY; j++) {
            cudaMemcpyAsync(d_input, vector+((i*EJEY + j)*ELEMS), ELEMS*sizeof(float), cudaMemcpyHostToDevice, stream);
            cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_input, d_output, ELEMS, 0, sizeof(float)*8, stream);
            cudaMemcpyAsync(output, d_output, ELEMS*sizeof(float), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
        }
    }
    free(output);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaStreamDestroy(stream);
    #pragma omp barrier
}
    gettimeofday(&tv_end, NULL);
    run_time=(tv_end.tv_sec - tv_start.tv_sec) * 1000000 + (tv_end.tv_usec - tv_start.tv_usec); //useconds
    run_time /= 1000000; // seconds
    printf("Tiempo: %lg segundos\n", run_time);
    printf("-----------------------------------\n");

    /* EJECUCION EN GPU CON LA FUNCION DEVICE RADIX SORT SIN STREAMS V2 */
    printf("Ejecucion CUB device radix sort sin streams v2\n");
    gettimeofday(&tv_start, NULL);
#pragma omp parallel 
{
    float* output;
    float* d_input;
    float* d_output;

    // Determine temporary device storage requirements
    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    output = (float*) malloc(EJEY*ELEMS*sizeof(float));
    cudaMalloc(&d_input, EJEY*ELEMS*sizeof(float));
    cudaMalloc(&d_output, EJEY*ELEMS*sizeof(float));

    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_input, d_output, ELEMS, 0, sizeof(float)*8);

    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    
    #pragma omp for
    for(int i = 0; i < EJEX; i++) {
        cudaMemcpy(d_input, vector+(i*EJEY*ELEMS), EJEY*ELEMS*sizeof(float), cudaMemcpyHostToDevice);
        for(int j = 0; j < EJEY; j++) {
            cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_input+(j*ELEMS), d_output+(j*ELEMS), ELEMS, 0, sizeof(float)*8);
            cudaDeviceSynchronize();
        }
        cudaMemcpy(output, d_output, EJEX*ELEMS*sizeof(float), cudaMemcpyDeviceToHost);
    }
    free(output);
    cudaFree(d_input);
    cudaFree(d_output);
    #pragma omp barrier
}
    gettimeofday(&tv_end, NULL);
    run_time=(tv_end.tv_sec - tv_start.tv_sec) * 1000000 + (tv_end.tv_usec - tv_start.tv_usec); //useconds
    run_time /= 1000000; // seconds
    printf("Tiempo: %lg segundos\n", run_time);
    printf("-----------------------------------\n");

    /* EJECUCION EN GPU CON LA FUNCION DEVICE RADIX SORT Y STREAMS V2 */
    printf("Ejecucion CUB device radix sort con streams v2\n");
    gettimeofday(&tv_start, NULL);
#pragma omp parallel 
{
    float* output;
    float* d_input;
    float* d_output;

    // Determine temporary device storage requirements
    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cudaStream_t stream;

    output = (float*) malloc(EJEY*ELEMS*sizeof(float));
    cudaMalloc(&d_input, EJEY*ELEMS*sizeof(float));
    cudaMalloc(&d_output, EJEY*ELEMS*sizeof(float));
    
    cudaStreamCreate(&stream);

    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_input, d_output, ELEMS, 0, sizeof(float)*8, stream);

    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    
    #pragma omp for
    for(int i = 0; i < EJEX; i++) {
        cudaMemcpyAsync(d_input, vector+(i*EJEY*ELEMS), EJEY*ELEMS*sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaStreamSynchronize(stream);
        for(int j = 0; j < EJEY; j++) {
            cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_input+(j*ELEMS), d_output+(j*ELEMS), ELEMS, 0, sizeof(float)*8, stream);
            cudaStreamSynchronize(stream);
        }
        cudaMemcpyAsync(output, d_output, EJEY*ELEMS*sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
    }
    free(output);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaStreamDestroy(stream);
    #pragma omp barrier
}
    gettimeofday(&tv_end, NULL);
    run_time=(tv_end.tv_sec - tv_start.tv_sec) * 1000000 + (tv_end.tv_usec - tv_start.tv_usec); //useconds
    run_time /= 1000000; // seconds
    printf("Tiempo: %lg segundos\n", run_time);
    printf("-----------------------------------\n");

    /* EJECUCION EN GPU CON LA FUNCION DEVICE SEGMENTED RADIX SORT SIN STREAMS */
    printf("Ejecucion CUB device segmented radix sort sin streams\n");
    gettimeofday(&tv_start, NULL);
#pragma omp parallel 
{
    float* output;
    float* d_input;
    float* d_output;
    int h_offsets[EJEY+1];
    int* d_offsets;

    // Determine temporary device storage requirements
    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    output = (float*) malloc(EJEY*ELEMS*sizeof(float));
    cudaMalloc(&d_input, EJEY*ELEMS*sizeof(float));
    cudaMalloc(&d_output, EJEY*ELEMS*sizeof(float));
    cudaMalloc(&d_offsets, (EJEY+1)*sizeof(int));

    for(int i = 0; i < (EJEY+1); i++) h_offsets[i] = ELEMS*i;

    cub::DeviceSegmentedRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_input, d_output, ELEMS, EJEY, 
                                            d_offsets, d_offsets+1, 0, sizeof(float)*8);

    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    cudaMemcpy(d_offsets, h_offsets, (EJEY+1)*sizeof(int), cudaMemcpyHostToDevice);

    #pragma omp for
    for(int i = 0; i < EJEX; i++) {
        cudaMemcpy(d_input, vector+(i*EJEY*ELEMS), EJEY*ELEMS*sizeof(float), cudaMemcpyHostToDevice);
        cub::DeviceSegmentedRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_input, d_output, ELEMS, EJEY, 
                                                d_offsets, d_offsets+1, 0, sizeof(float)*8);
        cudaMemcpy(output, d_output, EJEY*ELEMS*sizeof(float), cudaMemcpyDeviceToHost);
    }
    free(output);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_offsets);
    #pragma omp barrier
}
    gettimeofday(&tv_end, NULL);
    run_time=(tv_end.tv_sec - tv_start.tv_sec) * 1000000 + (tv_end.tv_usec - tv_start.tv_usec); //useconds
    run_time /= 1000000; // seconds
    printf("Tiempo: %lg segundos\n", run_time);
    printf("-----------------------------------\n");

    /* EJECUCION EN GPU CON LA FUNCION DEVICE RADIX SORT Y STREAMS V2 */
    printf("Ejecucion CUB device radix sort con streams v2\n");
    gettimeofday(&tv_start, NULL);
#pragma omp parallel 
{
    float* output;
    float* d_input;
    float* d_output;
    int h_offsets[EJEY+1];
    int* d_offsets;

    // Determine temporary device storage requirements
    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cudaStream_t stream;

    output = (float*) malloc(EJEY*ELEMS*sizeof(float));
    cudaMalloc(&d_input, EJEY*ELEMS*sizeof(float));
    cudaMalloc(&d_output, EJEY*ELEMS*sizeof(float));
    cudaMalloc(&d_offsets, (EJEY+1)*sizeof(int));
    cudaStreamCreate(&stream);

    for(int i = 0; i < (EJEY+1); i++) h_offsets[i] = ELEMS*i;

    cub::DeviceSegmentedRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_input, d_output, ELEMS, EJEY, 
                                            d_offsets, d_offsets+1, 0, sizeof(float)*8);

    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    cudaMemcpy(d_offsets, h_offsets, (EJEY+1)*sizeof(int), cudaMemcpyHostToDevice);

    #pragma omp for
    for(int i = 0; i < EJEX; i++) {
        cudaMemcpyAsync(d_input, vector+(i*EJEY*ELEMS), EJEY*ELEMS*sizeof(float), cudaMemcpyHostToDevice, stream);
        cub::DeviceSegmentedRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_input, d_output, ELEMS, EJEY, 
                                                d_offsets, d_offsets+1, 0, sizeof(float)*8, stream);
        cudaMemcpyAsync(output, d_output, EJEY*ELEMS*sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
    }
    free(output);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_offsets);
    cudaStreamDestroy(stream);
    #pragma omp barrier
}
    gettimeofday(&tv_end, NULL);
    run_time=(tv_end.tv_sec - tv_start.tv_sec) * 1000000 + (tv_end.tv_usec - tv_start.tv_usec); //useconds
    run_time /= 1000000; // seconds
    printf("Tiempo: %lg segundos\n", run_time);
    printf("-----------------------------------\n");

    free(vector);

    return 0;
}
