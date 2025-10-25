If you've read PMPP and looked through some GEMM implementations (lots of nice ones build up a modern GEMM in steps), but it still hasn't internalized, I hope my GEMM helps. Well actually... for simplicity we're going to start with a matmul:

```
C = A * B   // C=[M,N], A=[M,K], B=[K,N]
```

This is a classic CUDA tiled implementation, which I think should encapsulate the following:
- Tiling input/output matrices to avoid unnecessary loads from global memory
- Coalesced data loads from global memory into shared
- Loading data from shared memory to registers avoiding bank conflicts
- Accumulating output products into per-thread register tiles
- Evaluating occupancy from the code itself

And we don't worry about features like:
- Double buffering and async data loads from global memory
- Tensor core MMAs

Future TODO: Take this same GEMM, but do it with CuTe layouts.
Maybe TODO: Vectorized loads.