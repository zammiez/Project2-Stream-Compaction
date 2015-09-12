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
    // TODO use `thrust::exclusive_scan`
    // example: for device_vectors dv_in and dv_out:
    // thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());
	

	//thrust::host_vector<int> hst_in(n);
	//hst_in.resize(n);
	thrust::device_vector<int> dv_in (idata,idata+n) ;
	thrust::device_vector<int> dv_out = dv_in;


	//thrust::copy_n(idata, n, dv_in);

	thrust::exclusive_scan(dv_in.begin(), dv_in.end(), dv_out.begin());
	//thrust::exclusive_scan(idata, idata + n, odata);

	thrust::copy(dv_out.begin(), dv_out.end(), odata);

}

}
}
