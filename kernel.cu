/***
* SLIDING WINDOW SUM
* 
* launch a CUDA kernel that takes a vector of floats with a minimum of 5 elements, 
* applies a sliding window of 2 elements backwards and 2 forwards, calculates the average 
* of those elements and the current element and saves the result in the same index of the 
* current element but in another vector. The maximum size of the vector is the maximum 
* number of threads per block in X.
* 
* @Author: Braulio Solorio
* @Author: Tijash Salamanca
*/
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <ctime>

__global__ void transformToAverageVector(int* vecA, int* vecB, int size);
__host__ void transformToAverageVectorHost(int* vecA, int* vecB, int size);
__host__ void printVector(int* vec, int size, std::string);
__host__ int getMaxThreadNum();
__host__ bool cudaOperationSucced(std::string);

int main() {
	const int MAX_SIZE = getMaxThreadNum();
	if (MAX_SIZE == -1) return 1;

	int vecSize = 1;
	char showVectorsChar = 'y';

	std::cout << "======================== " << std::endl;
	std::cout << "| MIDTERM 1:           | " << std::endl;
	std::cout << "+----------------------| " << std::endl;
	std::cout << "| Duedate: feb 24 2025 | " << std::endl;
	std::cout << "|                      | " << std::endl;
	std::cout << "| Authors: Braulio S.  | " << std::endl;
	std::cout << "|          Tijash S.   | " << std::endl;
	std::cout << "======================== " << std::endl;
	std::cout << "Min vector size: 5" << std::endl;
	std::cout << "Max vector size: " << MAX_SIZE << std::endl;
	std::cout << "\nVector size\n[i]: ";
	std::cin >> vecSize;

	if (vecSize < 5) {
		std::cout << "Size must be in the range [5, " << MAX_SIZE << "]" << std::endl;
		return 1;
	}


	std::cout << "\nShow Final Vectors? (y/n)\n[i]: ";
	std::cin >> showVectorsChar;

	bool showVectors = showVectorsChar != 'n'; 
	

	int* vecA, * vecB;
	int* dev_vecA, * dev_vecB;

	vecA = (int*)malloc(sizeof(int) * vecSize);
	vecB = (int*)malloc(sizeof(int) * vecSize);

	cudaMalloc((void**)&dev_vecA, sizeof(int) * vecSize);
	if (!cudaOperationSucced("Cuda Malloc of dev_vecA")) return 1;

	 cudaMalloc((void**)&dev_vecB, sizeof(int) * vecSize);
	if (!cudaOperationSucced("Cuda Malloc of dev_vecB")) return 1;

	for (int i = 0; i < vecSize; i++) {
		vecA[i] = i + 1;
	}

	clock_t start = clock();
	transformToAverageVectorHost(vecA, vecB, vecSize);
	clock_t end = clock();
	if (showVectors) {
		std::cout << "CPU" << std::endl;
		printVector(vecA, vecSize, "Vector A");
		printVector(vecB, vecSize, "Vector B");
		std::cout << std::endl;
	}

	double CPU_TIME = ((double)(end - start)) / CLOCKS_PER_SEC;


	dim3 myGrid(1, 1, 1);
	dim3 myBlock(vecSize, 1, 1);


	cudaMemcpy(dev_vecA, vecA, sizeof(int) * vecSize, cudaMemcpyHostToDevice);
	if (!cudaOperationSucced("Cuda Memcpy of from vecA to dev_vecA")) return 1;

	start = clock();
	transformToAverageVector << <myGrid, myBlock >> > (dev_vecA, dev_vecB, vecSize);
	if (!cudaOperationSucced("transformToAverageVector in device")) return 1;

	cudaDeviceSynchronize();
	if (!cudaOperationSucced("Cuda Device Synchronize")) return 1;

	end = clock();
	double GPU_TIME = ((double)(end - start)) / CLOCKS_PER_SEC;

	cudaMemcpy(vecA, dev_vecA, sizeof(int) * vecSize, cudaMemcpyDeviceToHost);
	if (!cudaOperationSucced("Cuda Memcpy of from dev_vecA to vecA")) return 1;

	cudaMemcpy(vecB, dev_vecB, sizeof(int) * vecSize, cudaMemcpyDeviceToHost);
	if (!cudaOperationSucced("Cuda Memcpy of from dev_vecb to vecB")) return 1;

	if (showVectors) {
		std::cout << "GPU" << std::endl;
		printVector(vecA, vecSize, "Vector A");
		printVector(vecB, vecSize, "Vector B");
		std::cout << std::endl;
	}

	std::cout << "CPU: " << CPU_TIME << std::endl;
	std::cout << "GPU: " << GPU_TIME << std::endl;

	free(vecA);
	free(vecB);
	cudaFree(dev_vecA);
	if (!cudaOperationSucced("Cuda Free of dev_vecA")) return 1;
	cudaFree(dev_vecB);
	if (!cudaOperationSucced("Cuda Free of dev_vecB")) return 1;


	return 0;
}

/*
* Transform the values of vecB to the average of a window of 5 numbers from vecA. This function is executed on the device.
* Example:
*	@brief vecA = [1,2,3,4,5].
*	@brief vecB[2] = (1 + 2 + 3 + 4 + 5) / 5 = 3
*	@brief vecB[3] = (2 + 3 + 4 + 5 + 1) / 5 = 3
*   @brief vecB = [3,3,3,3,3,]
* 
* 
* @param vecA - a pointer to the first array of integers
* @param vecB - a pointer to the second array that will be transformed
* @param size - the size of the vectors.
* 
* @returns void
*/
__global__ void transformToAverageVector(int* vecA, int* vecB, int size) {
	int i = (blockDim.x * blockIdx.x + threadIdx.x);
	int idx = size + (i - 2);
	vecB[i] = (vecA[(idx + 0) % size] + vecA[(idx + 1) % size] + vecA[(idx + 2) % size] + vecA[(idx + 3) % size] + vecA[(idx + 4) % size]) / 5;
}

/*
* Transform the values of vecB to the average of a window of 5 numbers from vecA. This function is executed on the host
* Example:
*	@brief vecA = [1,2,3,4,5].
*	@brief vecB[2] = (1 + 2 + 3 + 4 + 5) / 5 = 3
*	@brief vecB[3] = (2 + 3 + 4 + 5 + 1) / 5 = 3
*   @brief vecB = [3,3,3,3,3,]
*
* @param vecA - a pointer to the first array of integers
* @param vecB - a pointer to the second array that will be transformed
* @param size - the size of the vectors.
*
* @returns void
*/
__host__ void transformToAverageVectorHost(int* vecA, int* vecB, int size) {
	for (int i = 0; i < size; i++) {
		int idx = size + (i - 2);
		vecB[i] = (vecA[(idx + 0) % size] + vecA[(idx + 1) % size] + vecA[(idx + 2) % size] + vecA[(idx + 3) % size] + vecA[(idx + 4) % size]) / 5;
	}
}

/*
* Displays a vector in console
* 
* @param vec - the vector to be displayed
* @param size - the size of the vector to be displayed
* @param name - A name tag to the vector that will be displayed in console
*/
__host__ void printVector(int* vec, int size, std::string name = "") {
	std::cout << name << ": ";
	for (int i = 0; i < size; i++) std::cout << vec[i] << " ";
	std::cout << std::endl;
}

/*
* Check the device properties and returns the max number of threads in the X dimension or -1 if an error occurred
* 
* @returns a integer representing the maximun number of threads in the X dimension or -1 if an error occurred
*/
__host__ int getMaxThreadNum() {
	cudaDeviceProp properties;
	cudaDeviceProp* properties_ptr = &properties;

	cudaGetDeviceProperties(properties_ptr, 0);

	int maxThreads = properties.maxThreadsDim[0];

	if (cudaOperationSucced("Reading device properties")) return maxThreads;

	return -1;
}

/*
* Gets the last cuda Error and if error is not cudaSuccess then prints it in console:
* 
* @param action - a description tag to know when the error occurred
* @returns if the cuda error is equal to cudaSuccess
*/
__host__ bool cudaOperationSucced(std::string action) {
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		std::cout << "[\033[31m" << "ERROR" << "\033[0m]" << " " << cudaGetErrorString(error) << ((action != "") ? " @ " : "") << action << std::endl;
		return false;
	}

	return true;
}