radix_f32: device_radix_sort_f32.cu common.h common.c
	nvcc -o radix_f32 device_radix_sort_f32.cu common.c -lm
radix_f64: device_radix_sort_f64.cu common.h common.c
	nvcc -o radix_f64 device_radix_sort_f64.cu common.c -lm
radix_i32: device_radix_sort_i32.cu common.h common.c
	nvcc -o radix_i32 device_radix_sort_i32.cu common.c -lm
radix_i16: device_radix_sort_i16.cu common.h common.c
	nvcc -o radix_i16 device_radix_sort_i16.cu common.c -lm
merge_f32: merge_sort_f32.cu common.h common.c
	nvcc -o merge_f32 merge_sort_f32.cu common.c -lm
merge_f64: merge_sort_f64.cu common.h common.c
	nvcc -o merge_f64 merge_sort_f64.cu common.c -lm
merge_i32: merge_sort_i32.cu common.h common.c
	nvcc -o merge_i32 merge_sort_i32.cu common.c -lm
merge_i16: merge_sort_i16.cu common.h common.c
	nvcc -o merge_i16 merge_sort_i16.cu common.c -lm
clean:
	rm -rf radix_??? merge_???