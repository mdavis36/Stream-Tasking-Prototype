#include <stdio.h>
#include <vector>
#include <array>
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

__global__ void calculation_kernel(float *x, int n)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
    x[i] = sqrt(pow(3.14159,i));
  }
}


//-----------------------------------------------------------------------------

template <int TASK_COUNT>
class TaskStreamGraph
{
  public:

    TaskStreamGraph(){
      for (int i = 0; i < TASK_COUNT; ++i){
	cudaStreamCreate(&streams[i]);
	cudaMalloc(&data[i], N * sizeof(float));
	cudaEventCreateWithFlags(&events[i], cudaEventDisableTiming);
      }
    }


    void dependentOn(int m, const std::vector<int> &n)
    {
      for (int i = 0; i < n.size(); ++i){
	if (dep_matrix[m][n[i]] != 1) {
	  dep_matrix[m][n[i]] = 1;
	  d_count[m]++;
	}
      }
    }


    std::vector<int> getDependents(int task){
      std::vector<int> results;
      for(int m = 0; m < TASK_COUNT; ++m){
	if (dep_matrix[m][task] == 1) results.push_back(m);
      }
      return results;
    }


    void buildAndExecuteGraph()
    {
      auto temp_d_count = d_count;
      exec_list.clear();
      pass_list.clear();

      do{
	pass_list.clear();

	for (int m = 0; m < TASK_COUNT; ++m){
	  if (temp_d_count[m] == 0){
	    pass_list.push_back(m);
	  }
	}

	for (int i = 0; i < pass_list.size(); ++i){
	  int n = pass_list[i];
	  for (int m = 0; m < TASK_COUNT; ++m){
	    if (dep_matrix[m][n] == 1){
	      temp_d_count[m]--;
	    }
	  }
	  temp_d_count[n] = -1;
	}
	exec_list.insert(exec_list.end(), pass_list.begin(), pass_list.end());

      }while(pass_list.size() > 0);

      for (int p : exec_list)
      {
	std::cout << p << " ";
	launch(p, getDependents(p));
      }
      std::cout << std::endl;
    }


    void print_dep_matrix(){
      for (int m = 0; m < TASK_COUNT; ++m){
	for (int n = 0; n < TASK_COUNT; ++n){
	  std::cout << dep_matrix[m][n] << " ";
	}
	std::cout << " - " << d_count[m] << std::endl;
      }
    }


  private:
    cudaStream_t streams[TASK_COUNT];
    cudaEvent_t events[TASK_COUNT];
    float *data[TASK_COUNT];
    int dep_matrix[TASK_COUNT][TASK_COUNT] = {0}; // [M][N] M is dependent on N
    std::array<int, TASK_COUNT> d_count = {0};

    std::vector<int> pass_list;
    std::vector<int> exec_list;


    void launch(int task, std::vector<int> dependants){
      calculation_kernel<<<1, 64, 0, streams[task]>>>(data[task], N);
      gpuErrchk(  cudaEventRecord(events[task], streams[task])  );

      for (int i = 0; i < dependants.size(); i++) {
	gpuErrchk(  cudaStreamWaitEvent(streams[dependants[i]], events[task], 0)  );
      }
    }

};


int main()
{
/*
  TaskStreamGraph<8> tsg;

  tsg.dependentOn(2, {0,1});
  tsg.dependentOn(3, {0,2});
  tsg.dependentOn(4, {3});
  tsg.dependentOn(5, {2});
  tsg.dependentOn(6, {4,5});
  tsg.dependentOn(7, {4,5});
*/

  TaskStreamGraph<16> tsg;

  tsg.dependentOn(1, {0});
  tsg.dependentOn(2, {3});
  tsg.dependentOn(4, {0});
  tsg.dependentOn(5, {1,4});
  tsg.dependentOn(6, {5,2,7});
  tsg.dependentOn(7, {3});
  tsg.dependentOn(8, {12});
  tsg.dependentOn(9, {8,13});
  tsg.dependentOn(10, {9,6,11});
  tsg.dependentOn(11, {15});
  tsg.dependentOn(13, {12});
  tsg.dependentOn(14, {15});

  tsg.print_dep_matrix();

  tsg.buildAndExecuteGraph();

  cudaDeviceReset();

  return 0;
    
}
