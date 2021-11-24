#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h> 
#include <cuda_runtime.h>

#define NUM_SM 40
#define WARP_SIZE 32
#define WARP_PER_BLOCK 4
#define BLOCK_PER_SM 8

#define FMA_USED true

#define HIGHEST_FREQ 1590000000ul
#define INSTRUCTION_PER_CLOCK 2
#define PEAK_PERF_FP32_PER_THREAD (HIGHEST_FREQ * 1.f * INSTRUCTION_PER_CLOCK)
#define PEAK_PERF_FP32 (PEAK_PERF_FP32_PER_THREAD * 16 * 4 * NUM_SM)

__device__ __inline__ float int_and_float_add(int tid) {
  register unsigned int num = 0;
  register unsigned int bound = HIGHEST_FREQ;
  register int local_tid = tid;
  register float res = 0.0f;
  for (int i = 0; i < INSTRUCTION_PER_CLOCK; i++) {
    num = 0;
    while (num < bound) {
      num = num + 1;
      res = res + num;
      // if (local_tid == 0 && num % 100000000 == 0) {
        // printf("bang %d!!!\n", num);
      // }
    }
  }
  return res;
}

__device__ __inline__ void fp32_mul() {
  // TODO(huangzan): how to set bound without a counter
  float num = 0.0f;
  float bound = PEAK_PERF_FP32_PER_THREAD;
  while (num < bound) {
    num = num * 1.1f;
  }
}

__global__ void cuda_kernel_launcher() {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  register float res = int_and_float_add(tid);
  if (tid == 0) {
    printf("res for thread 0 is %f\n", res);
  }
  __syncthreads();
}

void sass_kernel_launcher(char * file_name){

  int *output;
  cudaMalloc((void**)&output, sizeof(int));

  CUmodule module;
  CUfunction kernel;

  cuModuleLoad(&module, file_name);
  cuModuleGetFunction(&kernel, module, "kern");

  void * args[1] = {&output};

  cuLaunchKernel(kernel,
                 NUM_SM * BLOCK_PER_SM, 1, 1,
                 WARP_PER_BLOCK * WARP_SIZE, 1, 1,
                 4, 0, args, 0);
                 // 32 * 1024, 0, args, 0);

  int *output_h = (int*)malloc(sizeof(int));

  cudaMemcpy(output_h, output, sizeof(int), cudaMemcpyDeviceToHost);
  printf("%s took %d clocks.\n", file_name, output_h[0]);
  // printf("Each instruction takes %.2f clocks.\n\n", (float)output_h[0]/(128.0*128.0));

  cudaFree(output);
  free(output_h);

  printf("%f", output);
}

int main() {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  std::cout << "launching " << NUM_SM * BLOCK_PER_SM << " blocks,";
  std::cout << WARP_PER_BLOCK * WARP_SIZE << " threads per block" << std::endl;
  cudaEventRecord(start, 0);
  // cuda_kernel_launcher<<<NUM_SM * BLOCK_PER_SM, WARP_PER_BLOCK * WARP_SIZE>>>();
  sass_kernel_launcher("fp32ffma.cubin");
  cudaEventRecord(stop, 0);
  cudaStreamSynchronize(0);
  // cudaDeviceSynchronize();
  cudaEventSynchronize(stop);

  float elapsed_time;
  cudaEventElapsedTime(&elapsed_time, start, stop);

  double expected_time = 2 * WARP_SIZE * WARP_PER_BLOCK * NUM_SM * BLOCK_PER_SM * HIGHEST_FREQ / PEAK_PERF_FP32;  // reaching 8T instead of 4T flops depends on whether FMA instructions on/off
  std::cout << "time elapsed: " << elapsed_time/1000.f << " seconds" << std::endl;
  std::cout << "expected: " << expected_time  << " second" << std::endl;
  std::cout << ((FMA_USED)?1:2) * (1000 * expected_time) / elapsed_time << " of the peak performance achieved" << std::endl;
  std::cout << "peak performance is " << PEAK_PERF_FP32 << "fp32 flops" << std::endl;

  return 0;
}
