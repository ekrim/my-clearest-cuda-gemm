#include <cuda_runtime.h>

// thread block tiles of input matrices
constexpr int BM = 128;
constexpr int BK = 64;
constexpr int BN = 128;
// register tiles (each threads output tile)
constexpr int TM = 8;
constexpr int TN = 8;
// vectorized loads/stores
constexpr int numElemPerLoad = sizeof(uint4) / sizeof(__half); // =8 (load 16B worth of 2B __half's)
static_assert(numElemPerLoad == TN, "If we change register tile, a vectorized store does not perfectly cover a row of the tile");

template <int NumRowsInTile, int NumColsInTile, bool IsTransposed>
__device__ __forceinline__ void loadGlobalToShared(
	const __half *__restrict__ X, __half *__restrict__ Xs,
	int numColsGlobal, int rowGlobBase, int colGlobBase)
{
	constexpr int numVecLoads = (NumRowsInTile * NumColsInTile) / numElemPerLoad;
	const int threadID = threadIdx.y * blockDim.x + threadIdx.x;
	for (int i = threadID; i < numVecLoads; i += blockDim.x * blockDim.y) {
		// compute where we are in the tile by linear index -> tile row/col -> glob row/col -> linear global
	    const int linearIdxInTile = i * numElemPerLoad;
	    const int rowInTile = linearIdxInTile / NumColsInTile;
	    const int colInTile = linearIdxInTile % NumColsInTile;

	    const int rowGlob = rowGlobBase + rowInTile;
	    const int colGlob = colGlobBase + colInTile;

	    const int linearIdxGlob = rowGlob * numColsGlobal + colGlob; // X is row-major
	    const uint4 data = reinterpret_cast<const uint4*>(&X[linearIdxGlob])[0]; // 16B read to register

	    // store to shmem, transpose if needed
	    if constexpr (IsTransposed) {
			// non-vectorized store since we're storing contiguous chunk
			// See "Design Note: Bank Conflcits" in README.md
			__half* dataAsHalf = reinterpret_cast<__half*>(&data);
			#pragma unroll
			for (int j = 0; j < numElemPerLoad; ++j) {
			    Xs[colInBlock + j][rowInBlock] = dataAsHalf[j];
			}
	    } else {
	    	// vectorized store if we're taking 16B contiguous from global and putting them contiguous in shmem
		    reinterpret_cast<uint4*>(&Xs[rowInBlock][colInBlock])[0] = data;
	    }
	}
}

template <int NumColsInTile, int NumReg> // both As ans Bs have a 
__device__ __forceinline__ void loadSharedToRegisters(
	const __half (&Xs)[BK][NumColsInTile], __half (&h)[NumReg], )
{
    // Shared memory is 2-byte aligned per __half
    // To vector load 8 halfs (16 bytes), reinterpret pointer to uint4
    const uint4* vec_ptr = reinterpret_cast<const uint4*>(&Xs[row * 128 + col]);
    uint4 vec = *vec_ptr;

    // Reinterpret as 8 halfs
    __half* temp = reinterpret_cast<__half*>(&vec);

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        h[i] = temp[i];
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
	// this is 8*8*4B = 256B of register usage for the thread

	// zero init
    #pragma unroll
    for (int i = 0; i < Tm; ++i)
    #pragma unroll
    for (int j = 0; j < Tn; ++j)
        acc[i][j] = 0.0f;


	/* Step 2: Prepare shared memory tiles of A and B
	 * Store an A tile transposed: As[BK][BM]
	 *   So each thread will want a len-TM row segment of As for its outer prod (that slides down BK)
	 * Store a B tile as is: Bs[BK][BN]
	 *   Each thread still wants a row segment, len-TN, for the outer product.
	 *
	 * We want to read rows from shmem so data is contiguous and can use vectorized loads.
	 * For details about As vs Bs being transposed, see "Design Note: Bank Conflcits" in README.md
	 */
	 // TODO: padding?
	constexpr int padding = 
	__shared__ __align__(16) __half As[BK][BM];
	__shared__ __align__(16) __half Bs[BK][BN];

	// row and column upper left corner of this block
	const int rowGlobBase = blockIdx.y * blockDim.y;
	const int colGlobBase = blockIdx.x * blockDim.x;

	// Main outer loop moves the tiles across the common K dimension in BK strides 
	for (int idxKGlob = 0; idxKGlob < K; idxKGlob += BK) {

		// Step 3: Load into shmem from global
		loadGlobalToShared<True, BM, BK>(A, As, K, rowGlobBase, colGlobBase + idxKGlob);
		loadGlobalToShared<False, BK, BN>(B, Bs, N, rowGlobBase + idxKGlob, colGlobBase);
		__syncthreads(); // sync before trying to compute with these

		// Inner loop over the K dim of just the block
		// (each thread loads to registers and computes outer product)
        #pragma unroll
        for (int idxKBlock = 0; idxKBlock < BK; ++idxKBlock) {
			// Step 4: Load into registers from shmem
            float Areg[TM];
			loadSharedToRegisters<BM, TM>(As, Areg);
            float Breg[TN];
			loadSharedToRegisters<BN, TN>(Bs, Breg);

			// Step 5: Compute outer product
            // FMA into 8x8 fp32 accumulators
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
					acc[i][j] += Areg[i] * Breg[j];
		        }
            }
        }
		__syncthreads(); // sync before next tile
	}

	// Step 6: Each thread stores its 8x8 output tile starting at (global_row, global_col)
	#pragma unroll
	for (int i = 0; i < TM; ++i) {
	    __half h[TN];
	    #pragma unroll
	    for (int j = 0; j < TN; ++j) {
	        h[j] = __float2half(acc[i][j]);
	    }
		uint4 *Cvec = reinterpret_cast<uint4*>(&C[global_row * N + global_col]);
		uint4 accRow = reinterpret_cast<const uint4*>(h)[0];
		Cvec[i * (N / 8)] = accRow;
	}
}