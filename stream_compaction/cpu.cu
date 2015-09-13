#include <cstdio>
#include "cpu.h"
#include "common.h"
#include <cuda_runtime.h>
#include <ctime>

namespace StreamCompaction {
namespace CPU {

/**
 * CPU scan (prefix sum).
 */
void scan(int n, int *odata, const int *idata) {
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	std::clock_t t1, t2;
	t1 = std::clock();

	double duration;

	for (int i = 0; i < n; i++)
	{
		if (i==0)
		{
			odata[i] = 0;
			continue;
		}
		odata[i] = odata[i - 1] + idata[i - 1];
	}
	t2 = std::clock();

	duration = (float)t2 - (float)t1;

	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	printf("\t(CudaEvent)CPU time for scan : %3fms\n",milliseconds);
	printf("\t(Clock)CPU time for scan : %3fms\n", duration);

	// TO_DOne
    //printf("StreamCompaction::CPU::scan : exclusive prefix sum.\n");
}

/**
 * CPU stream compaction without using the scan function.
 *
 * @returns the number of elements remaining after compaction.
 */
int compactWithoutScan(int n, int *odata, const int *idata) {
	int k = 0;
	for (int i = 0; i < n; i++)
	{
		if (idata[i] != 0)
		{
			odata[k++] = idata[i];
		}
	}
    // TO__DOne
    return k;
}

/**
 * CPU stream compaction using scan and scatter, like the parallel version.
 *
 * @returns the number of elements remaining after compaction.
 */
int compactWithScan(int n, int *odata, const int *idata) {
    // TO_DOne
	int*TempArray = new int[n+1];
	int*ScanArray = new int[n+1];
	for (int i = 0; i < n; i++)
	{
		if (idata[i] != 0)
		{
			TempArray[i] = 1;
		}
		else
		{
			TempArray[i] = 0;
		}
	}
	TempArray[n] = 0;
	scan(n+1, ScanArray, TempArray);

	int k = 0;
	for (int i = 0; i < n; i++)
	{
		if (TempArray[i]==1)
		{
			odata[ScanArray[i]] =  idata[i];
			k++;
		}
	}
	int count =  ScanArray[n];
	delete[] TempArray;
	delete[] ScanArray;
	return count;
}

}
}
