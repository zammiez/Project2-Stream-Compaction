#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
	namespace Efficient {

		// TODO: __global__

		/**
		 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
		 */
		int * dev_o;

		__global__ void kernUpSweep(int pow2d, int * dev_odata,int end)
		{
			int k = blockIdx.x * blockDim.x + threadIdx.x;
			if (k > end) return;
			dev_odata[k * 2 * pow2d + (int)pow2d * 2 - 1] += dev_odata[k * 2 * pow2d + (int)pow2d - 1];
		}
		
		__global__ void kernDownSweep(int pow2d, int * dev_odata,int n,int end)
		{
			
			int k = (blockIdx.x * blockDim.x + threadIdx.x);
			if (k > end) return;
			k = k * 2 * pow2d;	
			//dev_odata[k * 2 * pow2d + (int)pow2d * 2 - 1] += dev_odata[k * 2 * pow2d + (int)pow2d - 1];
			int t = dev_odata[k + pow2d - 1];
			dev_odata[k + pow2d - 1] = dev_odata[k + pow2d * 2 - 1];
			dev_odata[k + pow2d * 2 - 1] += t;
		
		}
		__global__ void setRootZero(int * dev_odata,int n)
		{
			dev_odata[n - 1] = 0;
		}

		void initArrays(int n, const int *hst_idata)
		{
			int size = n*sizeof(int);

			cudaMalloc((void**)&dev_o, size);
			//checkCUDAError("cudaMalloc dev_o failed");
			cudaMemcpy(dev_o, hst_idata, size, cudaMemcpyHostToDevice);
			//checkCUDAError("cudaMemcpy odata->dev_o failed");

		}

		void freeArrays()
		{
			cudaFree(dev_o);
		}

		void scan(int n, int *odata, const int *idata) {
			// TO_DOne
			int N = ilog2ceil(n);
			N = pow(2,N);

			
			dim3 threadsPerBlock(blockSize_eff);

			initArrays(N, idata);

			cudaEvent_t start, stop;
			cudaEventCreate(&start);
			cudaEventCreate(&stop);

			cudaEventRecord(start);
			for (int d = 0; d <= ilog2ceil(N) - 1; d++)
			{
				int pow2d = pow(2, d);
				int end = (N - 1) / (2 * pow2d)+1;
				dim3 fullBlocksPerGrid((end + blockSize_eff - 1) / blockSize_eff);
				kernUpSweep <<<fullBlocksPerGrid, threadsPerBlock >>>(pow2d, dev_o, end);//later:blocksize,gridsize

			}
			setRootZero<<<1,1>>>(dev_o, N);
			for (int d = ilog2ceil(N) - 1; d >= 0; d--)
			{
				int pow2d = pow(2, d);
				int end = (N - 1) / (2 * pow2d) + 1;
				dim3 fullBlocksPerGrid((end + blockSize_eff - 1) / blockSize_eff);
				kernDownSweep <<<fullBlocksPerGrid, threadsPerBlock>>>(pow2d, dev_o,N,end);
			}
			cudaEventRecord(stop);
			cudaEventSynchronize(stop);
			float milliseconds = 0;
			cudaEventElapsedTime(&milliseconds, start, stop);

			cudaMemcpy(odata, dev_o, N*sizeof(int), cudaMemcpyDeviceToHost);

			printf("\t GPU time for efficient scan : %.4fms\n", milliseconds);
			
			freeArrays();
		}

		/**
		* Performs stream compaction on idata, storing the result into odata.
		* All zeroes are discarded.
		*
		* @param n      The number of elements in idata.
		* @param odata  The array into which to store elements.
		* @param idata  The array of elements to compact.
		* @returns      The number of elements remaining after compaction.
		*/
		int * dev_temp;
		int * dev_scan;
		int * dev_compactOut;
		int * dev_input;

		void freeCompArrays()
		{
			cudaFree(dev_temp);
			cudaFree(dev_scan);
			cudaFree(dev_compactOut);
			cudaFree(dev_input);
		}

		__global__ void kernMapToBoolean(int * dev_idata, int *dev_outTemp, int n)
		{
			int index = threadIdx.x;
			if (dev_idata[index] != 0 && index<n)
				dev_outTemp[index] = 1;
			else
				dev_outTemp[index] = 0;

		}
		__global__ void kernScatter(int *dev_t, int *dev_s, int *dev_in, int *dev_outCompact)
		{
			int index = threadIdx.x;
			if (dev_t[index] == 1)
			{
				dev_outCompact[dev_s[index]] = dev_in[index];
			}
		}

		int compact(int n, int *odata, const int *idata) {

			int N = ilog2ceil(n);
			N = pow(2, N);
			//int N = n;

			int size = N*sizeof(int);

			cudaMalloc((void**)&dev_temp, size);

			cudaMalloc((void**)&dev_input, size);
			cudaMemcpy(dev_input, idata, size, cudaMemcpyHostToDevice);

			cudaMalloc((void**)&dev_scan, size);


			//Step 1 : Compute temporary array.
			kernMapToBoolean << <1, N >> >(dev_input, dev_temp, n);//later:blocksize,gridsize 
			cudaMemcpy(dev_scan, dev_temp, size, cudaMemcpyDeviceToDevice);

			//Step 2 : Run exclusive scan

			for (int d = 0; d <= ilog2ceil(N) - 1; d++)
			{
				int pow2d = pow(2, d);
				int end = (N - 1) / (2 * pow2d) + 1;
				kernUpSweep << <1, end >> >(pow2d, dev_scan,end);//later:blocksize,gridsize
			}

			setRootZero << <1, 1 >> >(dev_scan, N);

			for (int d = ilog2ceil(N) - 1; d >= 0; d--)
			{
				int pow2d = pow(2, d);
				int end = (N - 1) / (2 * pow2d) + 1;
				kernDownSweep << <1, end >> >(pow2d, dev_scan, N,end);
			}

			//Step 3 : Scatter
			int compactLength;
			cudaMemcpy(&compactLength, &(dev_scan[N - 1]), sizeof(int), cudaMemcpyDeviceToHost);
			cudaMalloc((void**)&dev_compactOut, compactLength*sizeof(int));

			kernScatter << <1, N >> >(dev_temp, dev_scan, dev_input, dev_compactOut);//later:blocksize,gridsize

			//cudaMemcpy(odata,dev_compactOut,compactLength*sizeof(int),cudaMemcpyDeviceToHost);

			cudaMemcpy(odata, dev_compactOut, compactLength*sizeof(int), cudaMemcpyDeviceToHost);
			// TO_DOne
			freeCompArrays();
			return compactLength;
		}

	}
}