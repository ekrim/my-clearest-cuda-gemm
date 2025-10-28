#include <cuda_runtime.h>

// thread block tiles of input matrices
constexpr int BM = 128;
constexpr int BK = 64;
constexpr int BN = 128;
// register tiles (each threads output tile)
constexpr int TM = 8;
constexpr int TN = 8;
// vectorized loads/stores
constexpr int ONE_BANK_PADDING = 4 / sizeof(__half); // 4B/bank
constexpr int NUM_ELEM_PER_32BANKS = 32 * ONE_BANK_PADDING;
constexpr int NUM_ELEM_PER_LOAD = sizeof(uint4) / sizeof(__half); // =8 (load 16B worth of 2B __half's)
static_assert(NUM_ELEM_PER_LOAD == TN, "If we change register tile, a vectorized store does not perfectly cover a row of the tile");

static __device__ __forceinline__ int addPaddingToCol(int colWithoutPadding)
{
	return colWithoutPadding + ONE_BANK_PADDING * (colWithoutPadding / NUM_ELEM_PER_32BANKS);
}

template <int NumRowsInTile, int NumColsInTile, bool IsTransposed>
__device__ __forceinline__ void loadGlobalToShared(
	const __half *__restrict__ X, __half *__restrict__ Xs,
	int numColsGlob, int rowGlobTile, int colGlobTile)
{
	constexpr int numVecLoads = (NumRowsInTile * NumColsInTile) / NUM_ELEM_PER_LOAD;
	const int threadID = threadIdx.y * blockDim.x + threadIdx.x;
	for (int i = threadID; i < numVecLoads; i += blockDim.x * blockDim.y) {
		// compute where we are in the tile by linear index -> tile row/col -> glob row/col -> linear global
	    const int idxInTile = i * NUM_ELEM_PER_LOAD;  
		const int rowInTile = idxInTile / NumColsInTile;
	    const int colInTile = idxInTile % NumColsInTile;

		const int rowGlob = rowGlobTile + rowInTile;
		const int colGlob = colGlobTile + colInTile;
	    const int idxGlob = rowGlob * numColsGlob + colGlob; // X is row-major
	    const uint4 data = *reinterpret_cast<const uint4*>(&X[idxGlob]); // 16B read to register

	    // store to shmem, transpose if needed
	    if constexpr (IsTransposed) {
			// non-vectorized store since we're storing contiguous chunk as column in shmem
			// See "Design Note: Bank Conflicts" in README.md
			__half* dataAsHalf = reinterpret_cast<__half*>(&data);
			#pragma unroll
			for (int j = 0; j < NUM_ELEM_PER_LOAD; ++j) {
			    Xs[colInTile + j][addPaddingToCol(rowInTile)] = dataAsHalf[j];
			}
	    } else {
	    	// vectorized store if we're taking 16B contiguous from global and putting them contiguously into shmem
		    reinterpret_cast<uint4*>(&Xs[rowInTile][addPaddingToCol(colInTile)]) = *data;
	    }
	}
}

template <int NumReg>
__device__ __forceinline__ void loadSharedToRegisters(const __half *__restrict__ Xs, __half (&reg)[NumReg], int idxInTile)
{
	// vectorized load from shmem
    const uint4 data = *reinterpret_cast<const uint4*>(&Xs[idxInTile]);

    // reinterpret as 8 halfs
    const __half* dataAsHalf = reinterpret_cast<__half*>(&data);

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
	 * For details about As vs Bs being transposed and about padding see "Design Note: Bank Conflcits" in README.md
	 */
	// BM=BN=128 __halfs/row = 64 banks/row
	// put 1 bank's worth of padding after every 64 elements, so for each of the 4 subops of the v4 load
	// every thread in the warp is using a different bank
	__shared__ alignas(16) __half As[BK][BM + 2 * ONE_BANK_PADDING];
	__shared__ alignas(16) __half Bs[BK][BN + 2 * ONE_BANK_PADDING];

	// row and column (upper left origin) of this block in output mat
	const int rowGlobBlock = blockIdx.y * blockDim.y;
	const int colGlobBlock = blockIdx.x * blockDim.x;

	// Main outer loop moves the tiles across the common K dimension in BK strides 
	for (int kGlobTile = 0; kGlobTile < K; kGlobTile += BK) {

		// Step 3: Load into shmem from global
		loadGlobalToShared<True, BM, BK>(A, As, K, rowGlobBlock, colGlobBlock + kGlobTile); // tiles slide across A cols
		loadGlobalToShared<False, BK, BN>(B, Bs, N, rowGlobBlock + kGlobTile, colGlobBlock); // tiles slide down B rows
		__syncthreads(); // sync before trying to compute with these

		// Inner loop over the K dim of the tile
		// (each thread loads to registers and computes outer product)
        #pragma unroll
        for (int kInTile = 0; kInTile < BK; ++kInTile) {
			// Step 4: Load into registers from shmem
            // this block computes BMxBN output, but since we already loaded As[BK][BM] and Bs[BK][BN],
            // we just need tile-relative indices. For register-tile at [r][c] (relative to block-tile), we
            // take As[kInTile][r:r+TM] and Bs[kInTile][c:c+TN]
            float Areg[TM];
			const int rowInTileOfReg = threadIdx.y * TM;
			// Asmem is transposed, so to take a len-TM row segment, this thread's row acts like the col in the index calc below
			const int idxInTileAs = (kInTile * BM) + addPaddingToCol(rowInTileOfReg);
			loadSharedToRegisters<TM>(As, Areg, idxInTileAs);

            float Breg[TN];
			const int colInTileOfReg = threadIdx.x * TN;
			const int idxInTileBs = (kInTile * BN) + addPaddingToCol(colInTileOfReg);
			loadSharedToRegisters<TN>(Bs, Breg, idxInTileBs);

			// Step 5: Compute outer product
            // FMA into 8x8 fp32 accumulators
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
					acc[i][j] += Areg[i] * Breg[j]; // accumulate outer product for this BK-tile
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
	    __half alignas(16) row[TN];
	    #pragma unroll
	    for (int j = 0; j < TN; ++j) {
	        row[j] = __float2half(acc[i][j]);
	    }
		uint4 *dst = reinterpret_cast<uint4*>(&C[(rowGlobReg + i) * N + colGlobReg]);
		*dst = *reinterpret_cast<const uint4*>(row);
	}
}