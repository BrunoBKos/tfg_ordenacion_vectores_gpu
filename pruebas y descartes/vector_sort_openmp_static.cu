#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <math.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include "common.h"

/* THRESHOLD se refiere al número de elementos en un subarreglo.
   Si el subarreglo tiene menos elementos que el THRESHOLD, se ordena secuencialmente. De lo contrario,
   se divide en dos subarreglos y se crean tareas paralelas para ordenarlos.
   - Si el THRESHOLD es demasiado pequeño, se crean muchas tareas, lo que aumenta el overhead y puede saturar el sistema.
   - Si el THRESHOLD es demasiado grande, no se aprovecha el paralelismo disponible, y el rendimiento no mejora.
*/

int threshold;
int threads=1024;
int coef = 2; // Coeficiente de multiplicacion para determinar el THRESHOLD
int N=0;

///////////
// funciones auxiliares
///////////

int compare(const void *a, const void *b);
void swap(int *a, int *b);
int partition(int arr[], int low, int high);

void quicksort_parallel_sections(int arr[], int low, int high); //implementacion paralela OpenMP con "sections"
void quicksort_parallel_tasks(int arr[], int low, int high); //implementacion paralela OpenMP con "task

///////////
// funcion principal
///////////

int main(int argc, char *argv[]) {
    double start, end;
    srand(42);
    if (argc != 3) {
        printf("Uso: %s <N> <threads>\n", argv[0]);
        return 1;
    }
    N = atoi(argv[1]);
    threads = atoi(argv[2]);
    if (N <= 0 || threads <= 0) {
        printf("Error: N y threads deben ser mayores que 0.\n");
        return 1;
    }
    
    int* vector;
    int* output_vector;
    int* device_vector;
    
    vector = (int*) malloc(N*sizeof(int));
    output_vector = (int*) malloc(N*sizeof(int));
    cudaMalloc((void**) &device_vector, N * sizeof(int));

    initialize_vector_i32(vector, N);

    printf("----- Sort comparisons in CPU/GPU -----\n");
    printf("Max Threads:%d\n",omp_get_max_threads());
    printf("Threads:%d\n", threads);
    omp_set_num_threads(threads);
    printf("-----------------------------------\n");


    // USANDO THRUST CPU
    printf("Thrust CPU...\n");
    start = omp_get_wtime();
    memcpy(output_vector, vector, N*sizeof(int));
    thrust::sort(thrust::host, output_vector, output_vector + N);
    end = omp_get_wtime();
    printf("Tiempo: %f segundos\n", end - start);
    printf("-----------------------------------\n");    

    // USANDO THRUST GPU
    printf("Thrust GPU...\n");
    start = omp_get_wtime();
    cudaMemcpy(device_vector, vector, N*sizeof(int), cudaMemcpyHostToDevice);
    thrust::sort(thrust::device, device_vector, device_vector + N);
    cudaMemcpy(output_vector, device_vector, N*sizeof(int), cudaMemcpyDeviceToHost);
    end = omp_get_wtime();
    printf("Tiempo: %f segundos\n", end - start);
    printf("-----------------------------------\n");

    // PARALELO USANDO SECTIONS 
    printf("Paralelo OpenMP Sections...\n");
    start = omp_get_wtime();
    memcpy(output_vector, vector, N*sizeof(int));
    quicksort_parallel_sections(output_vector,0,N-1);
    end = omp_get_wtime();
    printf("Tiempo: %f segundos\n", end - start);
    printf("-----------------------------------\n");

    // PARALELO USANDO TASKS 
    printf("Paralelo OpenMP Tasks...\n");
    start = omp_get_wtime();
    memcpy(output_vector, vector, N*sizeof(int));
    quicksort_parallel_tasks(output_vector,0,N-1);
    end = omp_get_wtime();
    printf("Tiempo: %f segundos\n", end - start);
    printf("-----------------------------------\n");
    
    return 0;
}

int compare(const void *a, const void *b) {
    return (*(int*)a - *(int*)b);
}

void swap(int *a, int *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

/* Mediana de tres */
int median_of_three(int arr[], int low, int high) {
    int mid = low + (high - low) / 2;

    if (arr[low] > arr[mid]) swap(&arr[low], &arr[mid]);
    if (arr[low] > arr[high]) swap(&arr[low], &arr[high]);
    if (arr[mid] > arr[high]) swap(&arr[mid], &arr[high]);

    return mid; // Retorna índice del pivote
}

int partition(int arr[], int low, int high) { 
    /*-----------Estrategia de mediana de tres--------------*/ 
    int pivot_index = median_of_three(arr, low, high);
    swap(&arr[pivot_index], &arr[high]); // Mueve pivote al final
    /* -----------------------------------------------------*/
    int pivot = arr[high];  // Ultimo elemento como pivote
    int i = low - 1;  

    for (int j = low; j < high; j++) {
        if (arr[j] < pivot) {
            i++;
            swap(&arr[i], &arr[j]);
        }
    }
    swap(&arr[i + 1], &arr[high]);
    return (i + 1);
}

/* QUICKSORT PARALELO USANDO "SECTIONS" */
void quicksort_parallel_sections(int arr[], int low, int high) {
    threshold = N / (threads * pow(log2(threads), 1.25));  
    if (threshold < 20000) threshold = 20000;  
    
    if (low < high) {
        if (high - low < threshold || threads == 1) {
            qsort(arr + low, high - low + 1, sizeof(int), compare);
        } 
        else 
        {
            int pivot = partition(arr, low, high);
            if (high - low > threshold) {  // Evitar paralelismo en subarrays pequeños
                #pragma omp parallel
                {
                    #pragma omp sections
                    {
                        #pragma omp section
                        quicksort_parallel_sections(arr, low, pivot - 1);

                        #pragma omp section
                        quicksort_parallel_sections(arr, pivot + 1, high);
                    }
                }
            } 
            else {  
                // Si el subarray es pequeño, llamar recursivamente sin paralelismo
                quicksort_parallel_sections(arr, low, pivot - 1);
                quicksort_parallel_sections(arr, pivot + 1, high);
            }
        }
    }
}

/* QUICKSORT PARALELO USANDO "TASKS" */
void quicksort_parallel_tasks(int arr[], int low, int high) {
    threshold = N / (threads * sqrt(log2(threads) + 1));  
    if (threshold < 30000) threshold = 30000; 
    if (threads == 1) threshold = N;
    if (low < high) {
        if (high - low < threshold || threads == 1) {
            // Si el tamaño del subarray es menor que THRESHOLD, ya no usamos QuickSort recursivo, sino qsort (versión secuencial).
            qsort(arr + low, high - low + 1, sizeof(int), compare);
        } 
        else {
            int pivot = partition(arr, low, high);
            // high - low ≤ THRESHOLD, la recursión sigue ejecutándose normalmente pero en el mismo hilo en vez de crear nuevas tareas.
            #pragma omp task if (high - low > threshold)
            quicksort_parallel_tasks(arr, low, pivot - 1);
            
            #pragma omp task if (high - low > threshold)
            quicksort_parallel_tasks(arr, pivot + 1, high);

            //#pragma omp taskwait
            if (low == 0 && high == N - 1) {  
                #pragma omp taskwait  // Solo esperar en la primera llamada (root)
            }
        }
    }
}