#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include "common.h"
#include "thrust.h"

namespace StreamCompaction {
	namespace Thrust {

		/**
		* Performs prefix-sum (aka scan) on idata, storing the result into odata.
		*/
		void scan(int n, int *odata, const int *idata) {
			// TO_DOne use `thrust::exclusive_scan`
			// example: for device_vectors dv_in and dv_out:
			// thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());


			//thrust::host_vector<int> hst_in(n);
			//hst_in.resize(n);

			thrust::device_vector<int> dv_in(idata, idata + n);
			thrust::device_vector<int> dv_out = dv_in;


			//thrust::copy_n(idata, n, dv_in);
			cudaEvent_t start, stop;
			cudaEventCreate(&start);
			cudaEventCreate(&stop);

			cudaEventRecord(start);
			thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());
			cudaEventRecord(stop);

			cudaEventSynchronize(stop);
			float milliseconds = 0;
			cudaEventElapsedTime(&milliseconds, start, stop);
			printf("\t GPU time for thrust scan : %.4fms\n", milliseconds);
			thrust::copy(dv_out.begin(), dv_out.end(), odata);

		}

	}
	namespace RadixSort
	{
		__global__ void kernCalc_e(
			thrust::detail::normal_iterator<thrust::device_ptr<int>> dev_iArray,
			thrust::detail::normal_iterator<thrust::device_ptr<int>> dev_eArray,
			int n
			)
		{
			int index = blockIdx.x*blockDim.x + threadIdx.x;
			if (index >= n) return;
			int LSB = dev_iArray[index] % 2;
			dev_eArray[index] = (LSB == 1) ? 0 : 1;
		}
		__global__ void kernCalc_t(
			thrust::detail::normal_iterator<thrust::device_ptr<int>> dv_fArray,
			thrust::detail::normal_iterator<thrust::device_ptr<int>> dv_tArray,
			int totalFalses
			)
		{
			int index = threadIdx.x;

			dv_tArray[index] = index - dv_fArray[index] + totalFalses;
		}
		__global__ void kernCalc_d(
			thrust::detail::normal_iterator<thrust::device_ptr<int>> dv_fArray,
			thrust::detail::normal_iterator<thrust::device_ptr<int>> dv_tArray,
			thrust::detail::normal_iterator<thrust::device_ptr<int>> dv_eArray,
			thrust::detail::normal_iterator<thrust::device_ptr<int>> dv_dArray
			)
		{
			int index = threadIdx.x;
			bool bi = (dv_eArray[index] == 1) ? false : true;
			dv_dArray[index] = bi ? dv_tArray[index] : dv_fArray[index];
		}

		__global__ void kernScatterBasedOn_d(
			thrust::detail::normal_iterator<thrust::device_ptr<int>> dv_iArray,
			thrust::detail::normal_iterator<thrust::device_ptr<int>> dv_oArray,
			thrust::detail::normal_iterator<thrust::device_ptr<int>> dv_dArray
			)
		{
			int index = threadIdx.x;
			dv_oArray[dv_dArray[index]] = dv_iArray[index];
		}

		void sort(int n, int *odata, const int *idata)
		{
			//initArrays_RS(n, idata);
			thrust::device_vector<int> dv_i(idata, idata + n);
			thrust::device_vector<int> dv_e = dv_i;
			thrust::device_vector<int> dv_f = dv_e;
			thrust::device_vector<int> dv_t = dv_f;
			thrust::device_vector<int> dv_d = dv_f;
			thrust::device_vector<int> dv_o = dv_d;

			dim3 fullBlocksPerGrid((n + blockSize_thrust - 1) / blockSize_thrust);
			dim3 threadsPerBlock(blockSize_thrust);

			// Step 1 : Compute e array
			kernCalc_e <<<fullBlocksPerGrid, threadsPerBlock >>>(dv_i.begin(), dv_e.begin(), n);
			/*
			thrust::copy_n(dv_e.begin(), n, odata);
			printf("\n****** step 1 : dv_e\n----[");
			for (int i = 0; i < n; i++)
			{
				printf(" %3d", odata[i]);
			}
			printf("]\n"); //*/
			// Step 2 : Exclusive Scan e
			thrust::exclusive_scan(dv_e.begin(), dv_e.end(), dv_f.begin());
			/*
			thrust::copy_n(dv_f.begin(), n, odata);
			printf("\n****** step 2 : dv_f\n----[");
			for (int i = 0; i < n; i++)
			{
				printf(" %3d", odata[i]);
			}
			printf("]\n"); //*/

			//Step 3 : Compute totalFalses
			int totalFalses = dv_e[n - 1] + dv_f[n - 1];

			//Step 4 : Compute t array
			kernCalc_t << <1, n >> >(dv_f.begin(), dv_t.begin(), totalFalses);


			// Step 5 : Scatter based on d
			kernCalc_d << <1, n >> >(dv_f.begin(), dv_t.begin(), dv_e.begin(), dv_d.begin());


			kernScatterBasedOn_d << <1, n >> >(dv_i.begin(), dv_o.begin(), dv_d.begin());

			
			thrust::copy_n(dv_o.begin(), n, odata);
			/*printf("\n****** final : dv_o\n----[");
			for (int i = 0; i < n; i++)
			{
				printf(" %3d", odata[i]);
			}
			printf("]\n"); */

		}
	}
}