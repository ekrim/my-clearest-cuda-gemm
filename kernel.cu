#include <cuda_runtime.h>
#include <cuda_fp16.h>

// thread block tiles of input matrices
constexpr int BM = 128;
constexpr int BK = 64;
constexpr int BN = 128;
// register tiles (each threads output tile)
constexpr int TM = 8;
constexpr int TN = 8;
// vectorized loads/stores
constexpr int kOneBankPadding = 4 / sizeof(__half); // 4B/bank
constexpr int kNumElemPer32Banks = 32 * kOneBankPadding;
constexpr int kNumElemPerLoad = sizeof(uint4) / sizeof(__half); // =8 (load 16B worth of 2B __half's)
static_assert(kNumElemPerLoad == TN, "If we change register tile, a vectorized store does not perfectly cover a row of the tile");

static __device__ __forceinline__ int addPaddingToCol(int colWithoutPadding)
{
	return colWithoutPadding + kOneBankPadding * (colWithoutPadding / kNumElemPer32Banks);
}

template <int RowsSmem, int ColsSmem, int RowsInputTile, int ColsInputTile>
__device__ __forceinline__ void loadGlobalToShared(
	const __half *__restrict__ X, __half (&Xs)[RowsSmem][ColsSmem],
	int numColsGlob, int rowGlobTile, int colGlobTile)
{
	constexpr int kNumVecLoads = (RowsInputTile * ColsInputTile) / kNumElemPerLoad;
	const int threadID = threadIdx.y * blockDim.x + threadIdx.x;
	for (int i = threadID; i < kNumVecLoads; i += blockDim.x * blockDim.y) {
		// compute where we are in the tile by linear index -> tile row/col -> glob row/col -> linear global
	    const int idxInTile = i * kNumElemPerLoad;  
		const int rowInTile = idxInTile / ColsInputTile;
	    const int colInTile = idxInTile % ColsInputTile;

		const int rowGlob = rowGlobTile + rowInTile;
		const int colGlob = colGlobTile + colInTile;
	    const int idxGlob = rowGlob * numColsGlob + colGlob; // X is row-major
	    const uint4 data = *reinterpret_cast<const uint4*>(&X[idxGlob]); // 16B read to register

	    // store to shmem, transpose if needed
	    constexpr bool kIsTransposed = RowsSmem != RowsInputTile; // no padded rows, so we can use this check
	    if constexpr (kIsTransposed) {
			// non-vectorized store since we're storing contiguous chunk as column in shmem
			// See "Design Note: Memory" in README.md
			const __half* dataAsHalf = reinterpret_cast<const __half*>(&data);
			#pragma unroll
			for (int j = 0; j < kNumElemPerLoad; ++j) {
			    Xs[colInTile + j][addPaddingToCol(rowInTile)] = dataAsHalf[j];
			}
	    } else {
	    	// vectorized store if we're taking 16B contiguous from global and putting them contiguously into shmem
		    *reinterpret_cast<uint4*>(&Xs[rowInTile][addPaddingToCol(colInTile)]) = data;
	    }
	}
}

template <int NumReg>
__device__ __forceinline__ void loadSharedToRegisters(const __half *__restrict__ Xs, __half (&reg)[NumReg], int idxInTile)
{
	// vectorized load from shmem
    const uint4 data = *reinterpret_cast<const uint4*>(&Xs[idxInTile]);

    // reinterpret as 8 halfs
    const __half* dataAsHalf = reinterpret_cast<const __half*>(&data);

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        reg[i] = dataAsHalf[i];
    }
}

__global__ void gemmKernel(
	const __half* __restrict__ A,
	const __half* __restrict__ B,
	__half* __restrict__ C, int M, int N, int K)
{
	/* Step 1: Each thread is responsible for a TMxTN output tile of C:
     * Mechanism: if you want the 8x8 output with upper left corner at [r][c], you would take
	 *   for ik in range(0,BK):
	 *     Ctile += outerProd(A[r:r+8][ik], B[ik][c:c+8])
	 */ 
	float acc[TM][TN]; // results accumulate in fp32 as we iterate across the K dimension

	// zero init
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
	    #pragma unroll
	    for (int j = 0; j < TN; ++j) {
	        acc[i][j] = 0.f;
	    }
    }

	/* Step 2: Prepare shared memory tiles of A and B
	 * Store an A tile transposed: As[BK][BM]
	 *   So each thread will want a len-TM row segment of As for its outer prod (that slides down BK)
	 * Store a B tile as is: Bs[BK][BN]
	 *   Each thread still wants a row segment, len-TN, for the outer product.
	 *
	 * We want to read rows from shmem so data is contiguous and can use vectorized loads.
	 * For details about As vs Bs being transposed and about padding see "Design Note: Memory" in README.md
	 */
	// BM=BN=128 __halfs/row = 64 banks/row
	// put 1 bank's worth of padding after every 64 elements, so for each of the 4 subops of the v4 load
	// every thread in the warp is using a different bank
	 constexpr int kColsAs = BM + 2 * kOneBankPadding;
	 constexpr int kColsBs = BN + 2 * kOneBankPadding;
	__shared__ alignas(16) __half As[BK][kColsAs];
	__shared__ alignas(16) __half Bs[BK][kColsBs];

	// row and column (upper left origin) of this block in output mat
	const int rowGlobBlock = blockIdx.y * BM;
	const int colGlobBlock = blockIdx.x * BN;

	// Main outer loop moves the tiles across the common K dimension in BK strides 
	for (int kdimGlobTile = 0; kdimGlobTile < K; kdimGlobTile += BK) {

		// Step 3: Load into shmem from global
		loadGlobalToShared<BK, kColsAs, BM, BK>(A, As, K, rowGlobBlock, kdimGlobTile); // tiles slide across A cols
		loadGlobalToShared<BK, kColsBs, BK, BN>(B, Bs, N, kdimGlobTile, colGlobBlock); // tiles slide down B rows
		__syncthreads(); // sync before trying to compute with these

		// Inner loop over the K dim of the tile
		// (each thread loads to registers and computes outer product)
        #pragma unroll
        for (int kdimInTile = 0; kdimInTile < BK; ++kdimInTile) {
			// Step 4: Load into registers from shmem
            // this block computes BMxBN output, but since we already loaded As[BK][BM] and Bs[BK][BN],
            // we just need tile-relative indices. For register-tile at [r][c] (relative to block-tile), we
            // take As[kdimInTile][r:r+TM] and Bs[kdimInTile][c:c+TN]
            __half Areg[TM];
			const int rowInTileOfReg = threadIdx.y * TM;
			// Asmem is transposed, so to take a len-TM row segment, this thread's row acts like the col in the index calc below
			const int idxInTileAs = (kdimInTile * kColsAs) + addPaddingToCol(rowInTileOfReg);
			loadSharedToRegisters<TM>(&As[0][0], Areg, idxInTileAs);

            __half Breg[TN];
			const int colInTileOfReg = threadIdx.x * TN;
			const int idxInTileBs = (kdimInTile * kColsBs) + addPaddingToCol(colInTileOfReg);
			loadSharedToRegisters<TN>(&Bs[0][0], Breg, idxInTileBs);

			// Step 5: Compute outer product
            // FMA into 8x8 fp32 accumulators
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
					acc[i][j] += __half2float(Areg[i]) * __half2float(Breg[j]); // accumulate outer product for this BK-tile
		        }
            }
        }
		__syncthreads(); // sync before next tile
	}

	// Step 6: Store to global, this thread's TMxTN register tile
	// row and col (upper left origin) of the register tile
	const int rowGlobReg = rowGlobBlock + (threadIdx.y * TM);
	const int colGlobReg = colGlobBlock + (threadIdx.x * TN);

	#pragma unroll
	for (int i = 0; i < TM; ++i) {
	    __half row[TN];
	    #pragma unroll
	    for (int j = 0; j < TN; ++j) {
	        row[j] = __float2half(acc[i][j]);
	    }
		uint4 *dst = reinterpret_cast<uint4*>(&C[(rowGlobReg + i) * N + colGlobReg]);
		*dst = *reinterpret_cast<const uint4*>(row);
	}
}

int main()
{
	constexpr int M = 32 * BM;
	constexpr int K = 24 * BK;
	constexpr int N = 32 * BN;

	__half *hA, *hB, *hC;
	cudaMallocHost(&hA, M * K * sizeof(__half)); // pinned + aligned
	cudaMallocHost(&hB, K * N * sizeof(__half));
	cudaMallocHost(&hC, M * N * sizeof(__half));

	__half *dA, *dB, *dC;
	cudaMalloc(&dA, M * K * sizeof(__half));
	cudaMalloc(&dB, K * N * sizeof(__half));
	cudaMalloc(&dC, M * N * sizeof(__half));

	cudaMemcpy(dA, hA, M * K * sizeof(__half), cudaMemcpyHostToDevice);
	cudaMemcpy(dB, hB, K * N * sizeof(__half), cudaMemcpyHostToDevice);

	// launch
	dim3 block(16, 16, 1); // 256 threads per block
	dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM, 1);
	gemmKernel<<<grid, block>>>(dA, dB, dC, M, N, K);
	cudaDeviceSynchronize();

	// copy C to host
	cudaMemcpy(hC, dC, M * N * sizeof(__half), cudaMemcpyDeviceToHost);

	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dC);

	cudaFreeHost(hA);
	cudaFreeHost(hB);
	cudaFreeHost(hC);

	return 0;
}