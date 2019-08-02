#include<iostream>
#include<stdio.h>
#include<cuda_runtime.h>


__global__ void cuda_hello()
{
  printf("Hello from CUDA.\n");
}

int main()
{
  std::cout<<"Hello World"<<std::endl;
  cuda_hello<<<1,1>>>();
  return 0;
}
