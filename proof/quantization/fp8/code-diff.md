### 1. Inner-loop instruction: K-loop scale vs epilogue dequant

#### Naive — scale multiply inside every K-loop iteration

```cpp
float acc = 0.0f;
for (int tile = 0; tile < K; tile += TILE_K) {
    // load tiles to shared memory
    __syncthreads();
    for (int k = 0; k < TILE_K; ++k) {
        float a = float(As[ty][k]) * scaleA;   // × scaleA
        float b = float(Bs[k][tx]) * scaleB;   // × scaleB
        acc += a * b;
    }
    __syncthreads();
}
C[row * N + col] = acc;
```

**Two extra fp32 multiplies per inner-loop iteration.** For each of the `K / TILE_K` tiles, every element in the tile is multiplied by both `scaleA` and `scaleB` before the multiply-add. At TILE_K=16, this is 2 × 16 = 32 extra fp32 multiplies per output element per tile — all doing the same per-tensor scale factor.

The scale multiply is mathematically equivalent outside the loop (multiplication distributes over addition: `Σ(a·sA)·(b·sB) = sA·sB·Σ(a·b)`), so there is no numerical reason to keep it inside.

#### Skilled — scale applied once in epilogue

```cpp
float acc = 0.0f;
for (int tile = 0; tile < K; tile += TILE_K) {
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < TILE_K; ++k) {
        acc += float(As[ty][k]) * float(Bs[k][tx]);  // no scale
    }
    __syncthreads();
}
C[row * N + col] = acc * scaleA * scaleB;             // epilogue: 2 FMAs total
```

**Zero extra multiplies inside the loop.** The inner loop computes only the raw FP8 dot product. After all tiles are accumulated, a single multiply by `scaleA * scaleB` is applied in the epilogue — 2 FMAs per output element regardless of K.

The skill guidance: *"Apply per-tensor scale factors in the epilogue after all INT8 accumulation, not inside the tile loop. This removes 2×K/TILE_K extra FMAs per output element from the inner loop."*

---

### 2. Zero-scale guard: missing vs present

#### Naive — no guard, `amax=0` → NaN

```cpp
float compute_scale(const float* x, int n) {
    float amax = 0;
    for (int i = 0; i < n; ++i)
        amax = fmaxf(amax, fabsf(x[i]));
    return amax / 448.0f;           // amax=0 → 0.0
}
```

When the input is all-zeros, `amax = 0.0` and `compute_scale` returns `0.0 / 448.0 = 0.0`. The caller computes `invScale = 1.0 / scale = inf`, and the quantize kernel does `__nv_fp8_e4m3(0 * inf)` — producing `NaN` (E4M3 has no NaN encoding, but the CUDA conversion of `NaN` input via satfinite produces a vendor-specific bit pattern that reads back as `NaN`).

This affects any deployment where an input activation or weight tensor happens to be zero-filled — a zero-initialized bias, a masked-out attention head, or an empty sequence position.

#### Skilled — zero-scale guard, `amax=0` → clean

```cpp
float compute_scale(const float* x, int n) {
    float amax = 0;
    for (int i = 0; i < n; ++i)
        amax = fmaxf(amax, fabsf(x[i]));
    if (amax == 0.0f) return 1.0f;  // zero-scale guard
    return amax / 448.0f;
}
```

When `amax = 0`, the function returns `1.0` instead of `0.0`. The caller computes `invScale = 1.0 / 1.0 = 1.0`, and the quantize kernel produces `__nv_fp8_e4m3(0 * 1) = 0`. The GEMM then correctly computes `0 × 0 × scaleA × scaleB = 0`.

The skill guidance: *"Guard against zero-scale: when amax == 0, the scale should be 1.0, not 0. This prevents NaN from propagating through the quantize→GEMM→dequant pipeline on all-zero inputs."*

---

### 3. FP8 storage type: type-punned vs typed

#### Naive — `reinterpret_cast` from `unsigned char*`

```cpp
__nv_fp8_storage_t *dAq, *dBq;   // alias for unsigned char
// ... allocation and quantization ...
reinterpret_cast<__half*>(dAq)[i] = ...
```

The naive kernel uses `__nv_fp8_storage_t` (an alias for `unsigned char`) for device pointers and casts them to `__half*` for computation. This is undefined behavior in C++ — accessing an `unsigned char` object through a `__half*` pointer violates strict aliasing. The compiler may generate incorrect code, and the code assumes that `__nv_fp8_e4m3`, `unsigned char`, and `__half` have the same size (they do, but the assumption is fragile).

#### Skilled — properly typed `__nv_fp8_e4m3*`

```cpp
__nv_fp8_e4m3 *dAq, *dBq;          // native FP8 type
// ... natural element access ...
As[threadIdx.y][threadIdx.x] = A[row * K + a_col];
```

The skilled kernel uses the CUDA-provided `__nv_fp8_e4m3` type throughout — no casts, no aliasing violations. Shared memory tiles and device pointers all use the same type, and conversions to `float` or `half` for computation use either the built-in `operator float()` or explicit `__nv_cvt_fp8_to_halfraw` intrinsics.

The skill guidance: *"Always use `__nv_fp8_e4m3*` for FP8 device pointers. Avoid `reinterpret_cast` between FP8 storage types and computation types — use the explicit conversion intrinsics (`__nv_cvt_fp8_to_halfraw`, `__nv_cvt_float_to_fp8`) instead."*

---

### 4. Loop unrolling: compiler-default vs explicit

#### Naive — no unroll hint

```cpp
for (int k = 0; k < TILE_K; ++k) {
    float a = float(As[ty][k]) * scaleA;
    float b = float(Bs[k][tx]) * scaleB;
    acc += a * b;
}
```

The compiler is free to unroll or not. At TILE_K=16 with three fp32 operations per iteration, the compiler may choose a partial unroll factor. With `-O3`, `nvcc` usually unrolls small constant-trip-count loops, but without an explicit directive the behaviour is implementation-defined and may vary across CUDA versions.

#### Skilled — explicit `#pragma unroll`

```cpp
#pragma unroll
for (int k = 0; k < TILE_K; ++k) {
    acc += float(As[ty][k]) * float(Bs[k][tx]);
}
```

`#pragma unroll` tells the compiler to fully unroll the loop at TILE_K=16 — all 16 iterations become straight-line code. This eliminates loop overhead and gives the instruction scheduler more opportunities to hide latency by interleaving independent shared memory loads and FMAs.

The skill guidance: *"Use `#pragma unroll` on inner tile loops with a constant trip count. The compiler can then schedule loads from shared memory and FMA instructions across iterations, improving ILP."*
