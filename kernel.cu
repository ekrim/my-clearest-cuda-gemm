#include <cuda_runtime.h>

__global__ void gemm_kernel(
	const __half* __restrict__ A,
	const __half* __restrict__ B,
	__half* __restrict__ C, int M, int N, int K)
{
	// Step 1: Each thread is responsible for the following TMxTN output tile of C:
	static constexpr int TM = 8;
	static constexpr int TN = 8;
	/* Mechanism: if you want the 8x8 output with upper left corner at [r][c], you would take
	 *   for ik in range(0,BK):
	 *     Ctile += outerProd(A[r:r+8][ik], B[ik][c:c+8])
	 */ 
	float acc[TM][TN]; // results accumulate in fp32 as we iterate across the K dimension
	// this is 8*8*4B = 256B of register usage for the thread!

	// zero init
    #pragma unroll
    for (int i = 0; i < Tm; ++i)
    #pragma unroll
    for (int j = 0; j < Tn; ++j)
        acc[i][j] = 0.0f;

	// Step 2: Prepare shared memory tiles of A and B
	static constexpr int BM = 128;
	static constexpr int BK = 64;
	static constexpr int BN = 128;

	/* Shmem usage strategy:
	 * Store an A tile: As[BM][BK]
	 *   So each thread will want a len-TM column segment of A for its outer prod (that slides across BK)
	 * Store a B tile transposed: BsT[BN][BK]
	 *   So for B, it will still be a column segment, len-TN, for the outer product.
	 *
	 * Q: Are stride-BK accesses of shmem good?
	 *   Well in this case, numBanksPerRow = (BK * sizeof(__half)) / 4 bytes per bank = (64 * 2) / 4 = 32
	 *   Which meanss elements in a column are exactly 32 banks apart, and so the thread would have to 
	 */
	// so one row of our shmem, 
	static constexpr int padding = 
	__shared__ __half As[BM * BK]; 
	__shared__ __half BsT[BK * BN]; // we will store the transpose in shmem, so that reading 

	// Ste[]
}