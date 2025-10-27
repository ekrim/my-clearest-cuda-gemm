### Intro

If you've read PMPP and looked through some GEMM implementations (lots of nice ones build up a modern GEMM incrementally), but it still hasn't internalized, I hope mine helps. For simplicity we're going to start with a matmul:

```
C = A * B   // C=[M,N], A=[M,K], B=[K,N]
```

This is a classic CUDA tiled implementation, which IMO should encapsulate the following:
- Tiling input/output matrices to reuse data
- Coalesced and vectorized data loads from global memory into shared and from registers into global
- Minimizing bank conflicts, while also vectorizing data loads from shared memory to registers
- Accumulating output products into per-thread register tiles

And we don't worry about features like:
- Double buffering and async data loads from global memory
- Tensor core MMAs

Future TODO: Take this same GEMM, but do it with CuTe layouts.

### Design Note: Bank Conflicts

Each thread block works on a tile of `A` and `B` stored in shared memory. 