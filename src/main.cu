#include <stdio.h>
#include <vector>
#include <iostream>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
  if (code != cudaSuccess) 
  {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

const int N = 1 << 20;

__global__ void kernel(float *x, int n)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
    x[i] = sqrt(pow(3.14159,i));
  }
}

//-----------------------------------------------------------------------------

const int num_tasks = 8;
cudaStream_t streams[num_tasks];
cudaEvent_t events[num_tasks];
float *data[num_tasks];


__host__ void launch_task(int task, std::vector<int> dependants)
{
  kernel<<<1, 64, 0, streams[task]>>>(data[task], N);
  gpuErrchk(  cudaEventRecord(events[task], streams[task])  );

  for (int i = 0; i < dependants.size(); i++) {
    gpuErrchk(  cudaStreamWaitEvent(streams[dependants[i]], events[task], 0)  );
  }
}

int main()
{

  for (int i = 0; i < num_tasks; i++) {
    cudaStreamCreate(&streams[i]);
    cudaMalloc(&data[i], N * sizeof(float));
    cudaEventCreateWithFlags(&events[i], cudaEventDisableTiming);
  }

  launch_task(0, {2,3} );
  launch_task(1, {2}   );
  launch_task(2, {3,5} );
  launch_task(3, {4}   );
  launch_task(4, {6,7} );
  launch_task(5, {6,7} );
  launch_task(6, {}    );
  launch_task(7, {}    );
 
  cudaDeviceReset();

  return 0;
    
}
