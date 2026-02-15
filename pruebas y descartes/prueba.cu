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

#define EJEX 150
#define EJEY 200
#define ELEMS 27400

///////////
// Cabeceras
///////////

void cub_segmented_no_streams(float* vector, float* sorted_vector_par);

void cub_segmented_streams(float* vector, float* sorted_vector_par);

void mgpu_segmented(float* vector, float* sorted_vector_par);

void mgpu_segmented_streams(float* vector, float* sorted_vector_par);

void fast_segmented(float* vector, float* sorted_vector_par);

void cub_segmented_and_cpu(float* vector, float* sorted_vector_par);

void cub_segmented_streams_and_cpu(float* vector, float* sorted_vector_par);

///////////
// funcion principal
///////////

int main(int argc, char *argv[]) {

    struct timeval tv_start, tv_end;
    double run_time;
    
    float* vector;
    float* sorted_vector_sec;
    float* sorted_vector_par;

    printf("Inicio\n");
    gettimeofday(&tv_start, NULL);
    vector = (float*) malloc(EJEX*EJEY*ELEMS*sizeof(float));
    sorted_vector_sec = (float*) malloc(EJEX*EJEY*ELEMS*sizeof(float));
    sorted_vector_par = (float*) malloc(EJEX*EJEY*ELEMS*sizeof(float));
    // #pragma omp parallel for
    for(int i = 0; i < EJEX; i++) initialize_vector_f32(vector+(i*EJEY*ELEMS), EJEY*ELEMS);
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
    float* output;
    #pragma omp for
    for(int i = 0; i < EJEX; i++) {
        for(int j = 0; j < EJEY; j++) {
            output = (sorted_vector_sec+((i*EJEY + j)*ELEMS));
            memcpy(output, vector+((i*EJEY + j)*ELEMS), ELEMS*sizeof(float)); // copy of the original array
            thrust::sort(thrust::host, output, output+ELEMS); // sorting
        }
    }
    #pragma omp barrier
}
    gettimeofday(&tv_end, NULL);
    run_time=(tv_end.tv_sec - tv_start.tv_sec) * 1000000 + (tv_end.tv_usec - tv_start.tv_usec); //useconds
    run_time /= 1000000; // seconds
    printf("Tiempo: %lg segundos\n", run_time);
    printf("-----------------------------------\n");


    // /* EJECUCION EN GPU CON LA FUNCION DEVICE SEGMENTED RADIX SORT SIN STREAMS */
    // printf("Ejecucion CUB device segmented radix sort sin streams\n");
    // gettimeofday(&tv_start, NULL);
    // cub_segmented_no_streams(vector, sorted_vector_par);
    // gettimeofday(&tv_end, NULL);
    // run_time=(tv_end.tv_sec - tv_start.tv_sec) * 1000000 + (tv_end.tv_usec - tv_start.tv_usec); //useconds
    // run_time /= 1000000; // seconds
    // printf("Tiempo: %lg segundos\n", run_time);
    // printf("-----------------------------------\n"); 

    // /* EJECUCION EN GPU CON LA FUNCION DEVICE SEGMENTED RADIX SORT CON STREAMS */
    // printf("Ejecucion CUB device radix sort con streams v2\n");
    // gettimeofday(&tv_start, NULL);
    // cub_segmented_streams(vector, sorted_vector_par);
    // gettimeofday(&tv_end, NULL);
    // run_time=(tv_end.tv_sec - tv_start.tv_sec) * 1000000 + (tv_end.tv_usec - tv_start.tv_usec); //useconds
    // run_time /= 1000000; // seconds
    // printf("Tiempo: %lg segundos\n", run_time);
    // printf("-----------------------------------\n");

    /* EJECUCION EN GPU CON LA FUNCION SEGMENTED SORT DE MODERNGPU */
    // printf("Ejecucion Segmented Sort ModernGPUs sin streams\n");
    // gettimeofday(&tv_start, NULL);
    // mgpu_segmented(vector, sorted_vector_par);
    // gettimeofday(&tv_end, NULL);
    // run_time=(tv_end.tv_sec - tv_start.tv_sec) * 1000000 + (tv_end.tv_usec - tv_start.tv_usec); //useconds
    // run_time /= 1000000; // seconds
    // printf("Tiempo: %lg segundos\n", run_time);
    // printf("-----------------------------------\n");

    /* EJECUCION EN GPU CON LA FUNCION SEGMENTED SORT DE MODERNGPU CON STREAMS */
    // printf("Ejecucion Segmented Sort ModernGPUs con streams\n");
    // gettimeofday(&tv_start, NULL);
    // mgpu_segmented_streams(vector, sorted_vector_par);
    // gettimeofday(&tv_end, NULL);
    // run_time=(tv_end.tv_sec - tv_start.tv_sec) * 1000000 + (tv_end.tv_usec - tv_start.tv_usec); //useconds
    // run_time /= 1000000; // seconds
    // printf("Tiempo: %lg segundos\n", run_time);
    // printf("-----------------------------------\n");

    /* EJECUCION EN GPU CON LA FUNCION FAST SEGMENTED SORT */
    // printf("Ejecucion de Fast Segmented Sort\n");
    // gettimeofday(&tv_start, NULL);
    // fast_segmented(vector, sorted_vector_par);
    // gettimeofday(&tv_end, NULL);
    // run_time=(tv_end.tv_sec - tv_start.tv_sec) * 1000000 + (tv_end.tv_usec - tv_start.tv_usec); //useconds
    // run_time /= 1000000; // seconds
    // printf("Tiempo: %lg segundos\n", run_time);
    // printf("-----------------------------------\n");

    /* EJECUCION HYBRIDA CON CUB SEGMENTED SORT */
    // printf("Ejecucion de Fast Segmented Sort\n");
    // gettimeofday(&tv_start, NULL);
    // cub_segmented_and_cpu(vector, sorted_vector_par);
    // gettimeofday(&tv_end, NULL);
    // run_time=(tv_end.tv_sec - tv_start.tv_sec) * 1000000 + (tv_end.tv_usec - tv_start.tv_usec); //useconds
    // run_time /= 1000000; // seconds
    // printf("Tiempo: %lg segundos\n", run_time);
    // printf("-----------------------------------\n");

    /* EJECUCION HYBRIDA CON CUB SEGMENTED SORT */
    printf("Ejecucion de Fast Segmented Sort\n");
    gettimeofday(&tv_start, NULL);
    cub_segmented_streams_and_cpu(vector, sorted_vector_par);
    gettimeofday(&tv_end, NULL);
    run_time=(tv_end.tv_sec - tv_start.tv_sec) * 1000000 + (tv_end.tv_usec - tv_start.tv_usec); //useconds
    run_time /= 1000000; // seconds
    printf("Tiempo: %lg segundos\n", run_time);
    printf("-----------------------------------\n");


    int ret = compare_vector_f32(sorted_vector_sec, sorted_vector_par, EJEX*EJEY*ELEMS);

    int array_principal = ((ret+(EJEY*ELEMS)-1)/(EJEY*ELEMS));
    int subarray = (ret+ELEMS-1)/ELEMS;
    int elemento = ret%ELEMS;

    if(ret) printf("Ret: %d; Error en el array: %d, dentro del subarray: %d, en el elemento: %d\n",ret, array_principal, subarray, elemento);
    if(ret) {
        for(int i = elemento; i < ELEMS; i++) {
            printf("Expected: ");
            for(int j = 0; j < 6; j++)
                printf("%f ", sorted_vector_sec[ret+i]);
            printf("\nGiven: ");
            for(int j = 0; j < 6; j++)
                printf("%f ", sorted_vector_par[ret+i]);
            printf("\n");
        }
    }
    free(vector);
    free(sorted_vector_sec);
    free(sorted_vector_par);
    return 0;
}

void cub_segmented_no_streams(float* vector, float* sorted_vector_par) {
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

    cudaMalloc(&d_input, EJEY*ELEMS*sizeof(float));
    cudaMalloc(&d_output, EJEY*ELEMS*sizeof(float));
    cudaMalloc(&d_offsets, (EJEY+1)*sizeof(int));

    for(int i = 0; i < (EJEY+1); i++) h_offsets[i] = ELEMS*i;

    cub::DeviceSegmentedRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_input, d_output, ELEMS*EJEY, EJEY, 
                                            d_offsets, d_offsets+1, 0, sizeof(float)*8);

    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    cudaMemcpy(d_offsets, h_offsets, (EJEY+1)*sizeof(int), cudaMemcpyHostToDevice);

    #pragma omp for
    for(int i = 0; i < EJEX; i++) {
        cudaMemcpy(d_input, vector+(i*EJEY*ELEMS), EJEY*ELEMS*sizeof(float), cudaMemcpyHostToDevice);
        cub::DeviceSegmentedRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_input, d_output, ELEMS*EJEY, EJEY, 
                                                d_offsets, d_offsets+1, 0, sizeof(float)*8);
        output = (sorted_vector_par + (i*EJEY*ELEMS));
        cudaMemcpy(output, d_output, EJEY*ELEMS*sizeof(float), cudaMemcpyDeviceToHost);
    }
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_offsets);
    #pragma omp barrier
}
}

void cub_segmented_streams(float* vector, float* sorted_vector_par) {
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

    cudaMalloc(&d_input, EJEY*ELEMS*sizeof(float));
    cudaMalloc(&d_output, EJEY*ELEMS*sizeof(float));
    cudaMalloc(&d_offsets, (EJEY+1)*sizeof(int));
    cudaStreamCreate(&stream);

    for(int i = 0; i < (EJEY+1); i++) h_offsets[i] = ELEMS*i;

    cub::DeviceSegmentedRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_input, d_output, ELEMS*EJEY, EJEY, 
                                            d_offsets, d_offsets+1, 0, sizeof(float)*8);

    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    cudaMemcpy(d_offsets, h_offsets, (EJEY+1)*sizeof(int), cudaMemcpyHostToDevice);

    #pragma omp for
    for(int i = 0; i < EJEX; i++) {
        output = (sorted_vector_par + (i*EJEY*ELEMS));
        cudaMemcpyAsync(d_input, vector+(i*EJEY*ELEMS), EJEY*ELEMS*sizeof(float), cudaMemcpyHostToDevice, stream);
        cub::DeviceSegmentedRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_input, d_output, ELEMS*EJEY, EJEY, 
                                                d_offsets, d_offsets+1, 0, sizeof(float)*8, stream);
        cudaMemcpyAsync(output, d_output, EJEY*ELEMS*sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
    }
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_offsets);
    cudaStreamDestroy(stream);
    #pragma omp barrier
}

}

// Define a comparator functor
struct less_float {
    __device__ __host__ bool operator()(const float &a, const float &b) const {
        return a < b;
    }
};

void mgpu_segmented(float* vector, float* sorted_vector_par) {

    std::vector<int> h_offsets(EJEY+1);
    int* d_offsets;

    cudaMalloc(&d_offsets, (EJEY+1)*sizeof(int));

    for(int i = 0; i < (EJEY+1); i++) h_offsets[i] = ELEMS*i;
    cudaMemcpy(d_offsets, h_offsets.data(), (EJEY+1)*sizeof(int), cudaMemcpyHostToDevice);

    mgpu::standard_context_t context;

#pragma omp parallel 
{

    float* d_input;
    
    cudaMalloc(&d_input, EJEY*ELEMS*sizeof(float));

    #pragma omp for
    for(int i = 0; i < EJEX; i++) {
        cudaMemcpy(d_input, vector+(i*EJEY*ELEMS), EJEY*ELEMS*sizeof(float), cudaMemcpyHostToDevice);
        mgpu::segmented_sort(d_input, EJEY*ELEMS, d_offsets, EJEY, less_float(), context);
        cudaMemcpy(sorted_vector_par+(i*EJEY*ELEMS), d_input, EJEY*ELEMS*sizeof(float), cudaMemcpyDeviceToHost);
    }
    cudaFree(d_input);
    #pragma omp barrier
}
    cudaFree(d_offsets);
}


void mgpu_segmented_streams(float* vector, float* sorted_vector_par) {

    std::vector<int> h_offsets(EJEY+1);
    int* d_offsets;

    cudaMalloc(&d_offsets, (EJEY+1)*sizeof(int));

    for(int i = 0; i < (EJEY+1); i++) h_offsets[i] = ELEMS*i;
    cudaMemcpy(d_offsets, h_offsets.data(), (EJEY+1)*sizeof(int), cudaMemcpyHostToDevice);

#pragma omp parallel 
{

    float* d_input;
    cudaStream_t stream;
    
    cudaMalloc(&d_input, EJEY*ELEMS*sizeof(float));
    cudaStreamCreate(&stream);

    mgpu::standard_context_t context(false, stream);

    #pragma omp for
    for(int i = 0; i < EJEX; i++) {
        cudaMemcpyAsync(d_input, vector+(i*EJEY*ELEMS), EJEY*ELEMS*sizeof(float), cudaMemcpyHostToDevice, stream);
        mgpu::segmented_sort(d_input, EJEY*ELEMS, d_offsets, EJEY, less_float(), context);
        cudaMemcpyAsync(sorted_vector_par+(i*EJEY*ELEMS), d_input, EJEY*ELEMS*sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
    }
    cudaFree(d_input);
    cudaStreamDestroy(stream);

    #pragma omp barrier
}
    cudaFree(d_offsets);
}

void fast_segmented(float* vector, float* sorted_vector_par) {


    std::vector<int> h_offsets(EJEY+1);
    int* d_offsets;

    cudaMalloc(&d_offsets, (EJEY+1)*sizeof(int));

    for(int i = 0; i < (EJEY+1); i++) h_offsets[i] = ELEMS*i;
    cudaMemcpy(d_offsets, h_offsets.data(), (EJEY+1)*sizeof(int), cudaMemcpyHostToDevice);

//#pragma omp parallel 
//{
    float* d_input;
    float* d_values;
    
    cudaMalloc(&d_input, EJEY*ELEMS*sizeof(float));
    cudaMalloc(&d_values, EJEY*ELEMS*sizeof(float));

//    #pragma omp for
    for(int i = 0; i < EJEX; i++) {
        cudaMemcpy(d_input, vector+(i*EJEY*ELEMS), EJEY*ELEMS*sizeof(float), cudaMemcpyHostToDevice);
        bb_segsort<float, float>(d_input, d_values, EJEY*ELEMS, d_offsets, EJEY);
        cudaMemcpy(sorted_vector_par+(i*EJEY*ELEMS), d_input, EJEY*ELEMS*sizeof(float), cudaMemcpyDeviceToHost);
    }

    cudaFree(d_input);
    cudaFree(d_values);
//    #pragma omp barrier
//}
    cudaFree(d_offsets);
}

void cub_segmented_and_cpu(float* vector, float* sorted_vector_par) {
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

    if(omp_get_thread_num() == 0) {

        cudaMalloc(&d_input, EJEY*ELEMS*sizeof(float));
        cudaMalloc(&d_output, EJEY*ELEMS*sizeof(float));
        cudaMalloc(&d_offsets, (EJEY+1)*sizeof(int));

        for(int i = 0; i < (EJEY+1); i++) h_offsets[i] = ELEMS*i;

        cub::DeviceSegmentedRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_input, d_output, ELEMS*EJEY, EJEY, 
                                                d_offsets, d_offsets+1, 0, sizeof(float)*8);

        // Allocate temporary storage
        cudaMalloc(&d_temp_storage, temp_storage_bytes);

        cudaMemcpy(d_offsets, h_offsets, (EJEY+1)*sizeof(int), cudaMemcpyHostToDevice);
    }

    #pragma omp for
    for(int i = 0; i < EJEX; i++) {
        output = (sorted_vector_par + (i*EJEY*ELEMS));
        if(omp_get_thread_num() == 0) {
            cudaMemcpy(d_input, vector+(i*EJEY*ELEMS), EJEY*ELEMS*sizeof(float), cudaMemcpyHostToDevice);
            cub::DeviceSegmentedRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_input, d_output, ELEMS*EJEY, EJEY, 
                                                    d_offsets, d_offsets+1, 0, sizeof(float)*8);
            cudaMemcpy(output, d_output, EJEY*ELEMS*sizeof(float), cudaMemcpyDeviceToHost);
        } else {
            memcpy(output, vector+(i*EJEY*ELEMS), EJEY*ELEMS*sizeof(float)); // copy of the original array
            for(int j = 0; j < EJEY; j++) {
                thrust::sort(thrust::host, output, output+ELEMS); // sorting
                output += ELEMS;
            }
        }
    }
    if(omp_get_thread_num() == 0) {
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_offsets);
    }
    #pragma omp barrier
}

}

void cub_segmented_streams_and_cpu(float* vector, float* sorted_vector_par) {
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

    if(omp_get_thread_num() == 0 || omp_get_thread_num() == 1) {

        cudaMalloc(&d_input, EJEY*ELEMS*sizeof(float));
        cudaMalloc(&d_output, EJEY*ELEMS*sizeof(float));
        cudaMalloc(&d_offsets, (EJEY+1)*sizeof(int));
        cudaStreamCreate(&stream);

        for(int i = 0; i < (EJEY+1); i++) h_offsets[i] = ELEMS*i;

        cub::DeviceSegmentedRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_input, d_output, ELEMS*EJEY, EJEY, 
                                                d_offsets, d_offsets+1, 0, sizeof(float)*8);

        // Allocate temporary storage
        cudaMalloc(&d_temp_storage, temp_storage_bytes);

        cudaMemcpy(d_offsets, h_offsets, (EJEY+1)*sizeof(int), cudaMemcpyHostToDevice);
    }

    #pragma omp for
    for(int i = 0; i < EJEX; i++) {
        output = (sorted_vector_par + (i*EJEY*ELEMS));
        if(omp_get_thread_num() == 0 || omp_get_thread_num() == 1) {
            cudaMemcpyAsync(d_input, vector+(i*EJEY*ELEMS), EJEY*ELEMS*sizeof(float), cudaMemcpyHostToDevice, stream);
            cub::DeviceSegmentedRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_input, d_output, ELEMS*EJEY, EJEY, 
                                                    d_offsets, d_offsets+1, 0, sizeof(float)*8, stream);
            cudaMemcpyAsync(output, d_output, EJEY*ELEMS*sizeof(float), cudaMemcpyDeviceToHost, stream);
            cudaStreamSynchronize(stream);
        } else {
            memcpy(output, vector+(i*EJEY*ELEMS), EJEY*ELEMS*sizeof(float)); // copy of the original array
            for(int j = 0; j < EJEY; j++) {
                thrust::sort(thrust::host, output, output+ELEMS); // sorting
                output += ELEMS;
            }
        }
    }
    if(omp_get_thread_num() == 0 || omp_get_thread_num() == 1) {
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_offsets);
        cudaStreamDestroy(stream); 
    }
    #pragma omp barrier
}

}