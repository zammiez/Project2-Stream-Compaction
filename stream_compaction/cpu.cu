#include <cstdio>
#include "cpu.h"
#include "common.h"
namespace StreamCompaction {
namespace CPU {

/**
 * CPU scan (prefix sum).
 */
void scan(int n, int *odata, const int *idata) {
	for (int i = 0; i < n; i++)
	{
		if (i==0)
		{
			odata[i] = 0;
			continue;
		}
		odata[i] = odata[i - 1] + idata[i - 1];
	}
	// TO_DOne
    printf("StreamCompaction::CPU::scan : exclusive prefix sum.\n");
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
