cub: calculateCop_CUB.cu
	nvcc -O3 -o cub calculateCop_CUB.cu -lm -lnetcdf -Xcompiler -fopenmp -DBIG
cub_2: calculateCop_CUB_2.cu
	nvcc -O3 -o cub_2 calculateCop_CUB_2.cu -lm -lnetcdf -Xcompiler -fopenmp -DBIG
cub_batch: calculateCop_CUB_Batch.cu
	nvcc -O3 -o cub_batch calculateCop_CUB_Batch.cu -lm -lnetcdf -Xcompiler -fopenmp -DBIG
clean:
	rm -rf cub cub_2 cub_batch