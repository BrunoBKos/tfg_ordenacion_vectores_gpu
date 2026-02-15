#include <stddef.h>

void initialize_vector_i16(short* vect, size_t n);

void initialize_vector_i32(int* vect, size_t n);

void initialize_vector_f32(float* vect, size_t n);

void initialize_vector_f64(double* vect, size_t n);

int compare_vector_i16(short* vect_a, short* vect_b, size_t n);

int compare_vector_i32(int* vect_a, int* vect_b, size_t n);

int compare_vector_f32(float* vect_a, float* vect_b, size_t n);

int compare_vector_f64(double* vect_a, double* vect_b, size_t n);

int compare_i16(const void *p1, const void *p2);

int compare_i32(const void *p1, const void *p2);

int compare_f32(const void *p1, const void *p2);

int compare_f64(const void *p1, const void *p2);
