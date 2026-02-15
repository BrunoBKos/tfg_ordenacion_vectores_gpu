#include "common.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <string.h>


// auxiliar function to initialize vectors with random values (short)
void initialize_vector_i16(short* vect, size_t n) {
    for (int i = 0; i < n; i++) vect[i] = (short) (rand() % 1000);
}

// auxiliar function to initialize vectors with random values (int)
void initialize_vector_i32(int* vect, size_t n) {
    for (int i = 0; i < n; i++) vect[i] = (rand() % 1000);
}

/*
// auxiliar function to initialize vectors with random values (float)
void initialize_vector_f32(float* vect, size_t n) {
    for (int i = 0; i < n; i++) vect[i] = (float) (rand() % 1000);
}
*/

// auxiliar function to initialize vectors with random values (float)
void initialize_vector_f32(float* vect, size_t n) {
     srand((unsigned int) time(NULL));
     for (int i=0; i<n; i++) vect[i]=((float)rand()/(float) RAND_MAX)*1000.0;
}


/*// auxiliar function to initialize vectors with random values (float)
void initialize_vector_f32(float* vect, size_t n) {
    for (int i = 0; i < n; i++) vect[i] = (((float) (rand() % 1000)) / ((float) ((rand() % 1000)+1)));
}
*/

// auxiliar function to initialize vectors with random values (double)
void initialize_vector_f64(double* vect, size_t n) {
    for (int i = 0; i < n; i++) vect[i] = (((double) (rand() % 1000)) / ((double) ((rand() % 1000)+1)));
}

// auxiliar function to compare the values of two diferent vectors (short)
int compare_vector_i16(short* vect_a, short* vect_b, size_t n) {
    int i;
    for(i = 0; i < n; i++)
        if(vect_a[i] != vect_b[i]) break;
    return (i - n) ? i : 0;
}

// auxiliar function to compare the values of two diferent vectors (int)
int compare_vector_i32(int* vect_a, int* vect_b, size_t n) {
    int i;
    for(i = 0; i < n; i++)
        if(vect_a[i] != vect_b[i]) break;
    return (i - n) ? i : 0;
}

// auxiliar function to compare the values of two diferent vectors (float)
int compare_vector_f32(float* vect_a, float* vect_b, size_t n) {
    int i;
    for(i = 0; i < n; i++)
        if(vect_a[i] != vect_b[i]) break;
    return (i - n) ? i : 0;
}

// auxiliar function to compare the values of two diferent vectors (double)
int compare_vector_f64(double* vect_a, double* vect_b, size_t n) {
    int i;
    for(i = 0; i < n; i++)
        if(vect_a[i] != vect_b[i]) break;
    return (i - n) ? i : 0;
}

// Comparison function for qsort to sort in ascending order (short)
int compare_i16(const void *p1, const void *p2) {
    short short_a = *((short*)p1);
    short short_b = *((short*)p2);
    int res = 0;
    if(short_a < short_b) res--;
    if(short_a > short_b)  res++;
    return res;
}

// Comparison function for qsort to sort in ascending order (int)
int compare_i32(const void *p1, const void *p2) {
    int int_a = *((int*)p1);
    int int_b = *((int*)p2);
    int res = 0;
    if(int_a < int_b) res--;
    if(int_a > int_b)  res++;
    return res;
}

// Comparison function for qsort to sort in ascending order (float)
int compare_f32(const void *p1, const void *p2) {
    float float_a = *((float*)p1);
    float float_b = *((float*)p2);
    int res = 0;
    if(float_a < float_b) res--;
    if(float_a > float_b)  res++;
    return res;
}

// Comparison function for qsort to sort in ascending order (double)
int compare_f64(const void *p1, const void *p2) {
    double double_a = *((double*)p1);
    double double_b = *((double*)p2);
    int res = 0;
    if(double_a < double_b) res--;
    if(double_a > double_b)  res++;
    return res;
}
