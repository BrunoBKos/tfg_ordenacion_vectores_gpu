/*
 * Parallel bitonic sort using CUDA for multiple vectors
 * Based on: https://gist.github.com/mre/1392067
 * Adapted to sort 800 vectors of 28000 elements each
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include <float.h>

#define NUM_VECTORS 800
#define VECTOR_SIZE 28000
#define THREADS_PER_BLOCK 256

// Macro for checking CUDA errors
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

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

// Initialize vectors with random data
void initialize_vectors(float *vectors, int num_vectors, int vector_size) {
    srand(time(NULL));
    for (int i = 0; i < num_vectors; i++) {
        for (int j = 0; j < vector_size; j++) {
            vectors[i * vector_size + j] = (float)rand() / RAND_MAX * 10000.0f;
        }
    }
}

// Verify that all vectors are sorted
// ATN: modified datsi
//int verify_sorting(float *vectors, int num_vectors, int vector_size) {
int verify_sorting(float *vectors, int num_vectors, int vector_size, int padded_size) {
    int errors = 0;
    // printf("FLT_MAX = %.2f\n", FLT_MAX);
    for (int i = 0; i < num_vectors; i++) {
      //        for (int j = 0; j < vector_size - 1; j++) {
        for (int j = 0; j < vector_size - 2; j++) {
	  //            if (vectors[i * vector_size + j] > vectors[i * vector_size + j + 1]) {
            if (vectors[i * padded_size + j] > vectors[i * padded_size + j + 1]) {
                if (errors < 10) { // Only print first 10 errors
                    printf("Error: Vector %d not sorted at position %d\n", i, j);
                    printf("  values[%d] = %.2f, values[%d] = %.2f\n", 
			   //                           j, vectors[i * vector_size + j], 
			   //                           j + 1, vectors[i * vector_size + j + 1]);
                           j, vectors[i * padded_size + j], 
                           j + 1, vectors[i * padded_size + j + 1]);
                }
                errors++;
            }
        }
    }
    return errors == 0;
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
            
            // Check for kernel launch errors
            CUDA_CHECK(cudaGetLastError());
        }
    }
    
    // Synchronize to ensure all kernels have completed
    CUDA_CHECK(cudaDeviceSynchronize());
}

int main() {
    // Calculate padded size (bitonic sort requires power of 2)
    int padded_size = next_power_of_2(VECTOR_SIZE);
    
    printf("=== CUDA Bitonic Sort for Multiple Vectors ===\n");
    printf("Original vector size: %d\n", VECTOR_SIZE);
    printf("Padded vector size: %d (next power of 2)\n", padded_size);
    printf("Number of vectors: %d\n", NUM_VECTORS);
    printf("Total elements: %d\n\n", NUM_VECTORS * padded_size);
    
    // Allocate host memory
    size_t data_size = NUM_VECTORS * padded_size * sizeof(float);
    float *h_vectors = (float*)malloc(data_size);
    
    if (h_vectors == NULL) {
        fprintf(stderr, "Host memory allocation failed\n");
        return 1;
    }
    
    // Initialize vectors with random data
    printf("Initializing vectors with random data...\n");
    initialize_vectors(h_vectors, NUM_VECTORS, VECTOR_SIZE);
    
    // Pad vectors with maximum float value (so they sort to the end)
    for (int i = 0; i < NUM_VECTORS; i++) {
        for (int j = VECTOR_SIZE; j < padded_size; j++) {
	  //            h_vectors[i * padded_size + j] = FLT_MAX;
	  h_vectors[i * padded_size + j] = FLT_MAX;
        }
    }
    
    // Print sample of unsorted data
    printf("\nFirst 10 elements of vector 0 (before sort):\n");
    for (int i = 0; i < 10; i++) {
        printf("%.2f ", h_vectors[i]);
    }
    printf("\n");
    
    // Allocate device memory
    float *d_vectors;
    CUDA_CHECK(cudaMalloc(&d_vectors, data_size));
    
    // Copy data to device
    printf("\nCopying data to GPU...\n");
    CUDA_CHECK(cudaMemcpy(d_vectors, h_vectors, data_size, cudaMemcpyHostToDevice));
    
    // Get device properties
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s\n", prop.name);
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // Launch sorting
    printf("\nLaunching bitonic sort...\n");
    CUDA_CHECK(cudaEventRecord(start));
    
    bitonic_sort(d_vectors, NUM_VECTORS, padded_size);
    
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    // Calculate elapsed time
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Sorting completed in %.3f ms\n", milliseconds);
    
    // Copy results back to host
    printf("Copying results back to CPU...\n");
    CUDA_CHECK(cudaMemcpy(h_vectors, d_vectors, data_size, cudaMemcpyDeviceToHost));
    
    // Verify results (only check original vector size, not padding)
    printf("\nVerifying sorting...\n");
    //    if (verify_sorting(h_vectors, NUM_VECTORS, VECTOR_SIZE)) {
    if (verify_sorting(h_vectors, NUM_VECTORS, VECTOR_SIZE, padded_size)) {
        printf(" SUCCESS: All %d vectors sorted correctly!\n\n", NUM_VECTORS);
    } else {
        printf(" FAILURE: Some vectors are not sorted correctly!\n\n");
    }
    
    // Print sample results from first vector
    printf("First 10 elements of vector 0 (after sort):\n");
    for (int i = 0; i < 10; i++) {
        printf("%.2f ", h_vectors[i]);
    }
    printf("\n\nLast 10 elements of vector 0 (excluding padding):\n");
    for (int i = VECTOR_SIZE - 10; i < VECTOR_SIZE; i++) {
        printf("%.2f ", h_vectors[i]);
    }
    printf("\n");
    
    // Print sample from another vector to verify
    printf("\nFirst 10 elements of vector 799 (after sort):\n");
    for (int i = 0; i < 10; i++) {
        printf("%.2f ", h_vectors[799 * padded_size + i]);
    }
    printf("\n");
    
    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_vectors));
    free(h_vectors);
    
    printf("\nProgram completed successfully!\n");
    return 0;
}
