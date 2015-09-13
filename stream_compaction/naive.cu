#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
	namespace Naive {

		// TODO: __global__

		/**
		 * Performs prefix-sum (aka scan) on idata, storing the result into odata.
		 */
		int * dev_o;

		__global__ void kernNaiveScan(int pow2d_1, int *dev_odata)
		{
			int k = blockIdx.x * blockDim.x + threadIdx.x;
			if (k >= pow2d_1)
				dev_odata[k] = dev_odata[k - pow2d_1] + dev_odata[k];
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
			dim3 threadsPerBlock(blockSize_naive);
			dim3 fullBlocksPerGrid((n + blockSize_eff - 1) / blockSize_eff);
			//??? inclusive or exclusive ? not exactly 39.2/slides
			initArrays(n, idata);

			cudaEvent_t start, stop;
			cudaEventCreate(&start);
			cudaEventCreate(&stop);

			cudaEventRecord(start);
			for (int d = 1; d <= ilog2ceil(n); d++)
			{
				int pow2_d_1 = pow(2, d - 1);
				kernNaiveScan <<<fullBlocksPerGrid, threadsPerBlock >>>(pow2_d_1, dev_o); //later:blocksize,gridsize
			}
			cudaEventRecord(stop);
			cudaEventSynchronize(stop);
			float milliseconds = 0;
			cudaEventElapsedTime(&milliseconds, start, stop);
			printf("\t GPU time for naive scan : %.4fms\n",milliseconds);
			// TO_DOne
			cudaMemcpy(odata, dev_o, n*sizeof(int), cudaMemcpyDeviceToHost);
			//inclusive to exclusive
			for (int i = n-1; i >0; i--)
			{
				odata[i] = odata[i - 1];
			}
			odata[0] = 0;
			freeArrays();
		}

	}
}
