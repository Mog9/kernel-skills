### 1. Inner-loop instruction: scalar vs `dp4a`

#### Naive — scalar INT8 multiply-add

```cpp
int32_t acc = 0;
for (int k = 0; k < K; ++k)
    acc += (int32_t)A[row * K + k] * (int32_t)B[k * N + col];
```

**One multiply and one add per K-element.** Each iteration loads two INT8 values, extends to INT32, multiplies, and accumulates. For K=128, this is 128 multiply-adds per output element. The compiler emits separate `IADD` and `IMUL` instructions — no special INT8-dot-product hardware is used.

No constraint on K — works for any value. But the scalar approach uses 4× the instructions of `dp4a` for the same computation, leaving INT8 throughput on the table.

#### Skilled — `dp4a` packed INT8 dot product

```cpp
for (int k = 0; k < K; k += 4) {
    // Load 4 consecutive INT8 from A — single 32-bit load
    int32_t a4 = *reinterpret_cast<const int32_t*>(a_row + k);

    // Pack 4 strided INT8 from B column into one int32
    int32_t b4;
    int8_t* bp = reinterpret_cast<int8_t*>(&b4);
    bp[0] = B[(k+0)*N + col];
    bp[1] = B[(k+1)*N + col];
    bp[2] = B[(k+2)*N + col];
    bp[3] = B[(k+3)*N + col];

    // Single dp4a: 4 muls + 3 adds in one instruction
    acc = __dp4a(a4, b4, acc);
}
```

**Four multiply-adds in one instruction.** The `__dp4a` intrinsic emits a single PTX `dp4a` instruction that computes `dot4(int8[4], int8[4])` and accumulates into INT32. For K=128, this is 32 iterations instead of 128 — 4× fewer loop iterations.

The skill (§38, §62): *"The `dp4a` instruction computes `int32 += dot(int8[4], int8[4])` — four INT8 multiplications and an INT32 accumulation in a single instruction."*

The cost is that K must be a multiple of 4, and B loads are strided (non-contiguous column access). The A load is contiguous (row-major), and the 4 consecutive INT8 values are loaded as a single `int32` — trivial for dense row-major A. The B column loads are strided by N, requiring manual packing into the `b4` register.

---

### 2. Kernel structure: 4 launches vs 3 launches

#### Naive — three separate kernels, four launches

```
launch 1: quantize_kernel(A_fp32 → A_int8)
launch 2: quantize_kernel(B_fp32 → B_int8)
launch 3: int8_gemm_kernel(A_int8, B_int8 → C_int32)
launch 4: dequantize_kernel(C_int32 → C_fp32)
```

**INT32 accumulator written to global memory, then dequantized in a separate pass.** The GEMM kernel outputs `int32_t* C`, which is written to global memory and read back by the dequantize kernel. This adds:
- One global memory write of the full C matrix (M×N × 4 bytes)
- One global memory read of the full C matrix
- One extra kernel launch overhead (~5–10 µs)

For small matrices this overhead is significant relative to compute time.

#### Skilled — fused GEMM + dequant epilogue, three launches

```
launch 1: quantize_kernel(A_fp32 → A_int8)
launch 2: quantize_kernel(B_fp32 → B_int8)
launch 3: int8_gemm_dp4a_kernel(A_int8, B_int8 → C_fp32)
```

**Dequantization fused into the GEMM epilogue.** After the INT32 accumulation loop, the scale multiply is applied before storing:

```cpp
C[row * N + col] = (float)acc * combined_scale;
```

No intermediate `int32_t* C` buffer in global memory. The skill (§55): *"Apply scale and zero-point corrections in fp32 in the epilogue, after all INT32 accumulation is complete."*

Saves 1 kernel launch and 2 global memory round-trips for M×N elements. For the benchmark shapes, this contributes ~10–20% of the total speedup.

---

### 3. Overflow safety: missing vs verified

#### Naive — no overflow check

The naive kernel assumes INT32 accumulation is safe for all K. At K=128 with values in [-127, 127], the worst-case accumulator is `127 × 127 × 128 = 2,064,512` — well under `INT32_MAX`. But for K > ~133,169 or with larger quantization ranges, overflow is possible. The kernel silently produces wrong results with no indication.

#### Skilled — explicit overflow verification

```cpp
long long worst = 127LL * 127LL * K;
if (worst >= INT32_MAX) {
    fprintf(stderr, "Overflow risk!\n");
    return 1;
}
```

The skill (§36): *"Maximum value before overflow: `127 * 127 * K`. For K=4096: max = `127 * 127 * 4096 = ~66M`, which fits in INT32 (max ~2.1B). For K >> 65536, overflow risk exists."*

The check is a quick host-side computation before any kernel launches. It catches the degenerate case and prevents silent data corruption. Combined with the skill's suggestion to use per-channel quantization when K is large (which reduces per-dot-product K), this is a cheap but important safety net.

---

### 4. Accuracy metrics: basic vs comprehensive

#### Naive — unguarded relative error

```cpp
float re = (fabsf(ref[i]) > 1e-9f) ? ae / fabsf(ref[i]) : ae;
```

**Unfiltered relative error explodes for near-zero reference values.** When `ref[i]` is tiny (e.g., 1e-8) and `ae` is moderate (e.g., 1e-4), the relative error is `1e-4 / 1e-8 = 10,000%` — but this is just quantization noise, not a real accuracy problem. The naive kernel reports `max_relative_error = 2291%` for the test case, which looks catastrophic but is actually meaningless.

#### Skilled — filtered relative error + SNR

```cpp
float thresh = amax_ref * 0.01f;   // 1% of dynamic range
if (fabsf(ref[i]) >= thresh) {
    float re = ae / fabsf(ref[i]);
    max_re_filtered = fmaxf(max_re_filtered, re);
}
```

**Filters out near-zero reference values.** The threshold at 1% of the dynamic range prevents division-by-tiny artifacts. The same test case produces `max_rel_error = 31.6%` — representing the actual worst-case significant error. SNR is also reported (45.14 dB for the test case), giving a meaningful single-number quality metric.

---

### 5. K alignment handling: unchecked vs enforced

#### Naive — no K constraint

The naive scalar loop works for any K. There is no check on whether K is a multiple of anything. This is flexible but means the kernel cannot use `dp4a` — the scalar path is the only option regardless of K.

#### Skilled — K must be multiple of 4

```cpp
if (K % 4 != 0) {
    fprintf(stderr, "K=%d is not a multiple of 4. Pad K before calling the kernel.\n", K);
    return 1;
}
```

The skill (§57, §67): *"K must be a multiple of 4 for `dp4a`. If K is not a multiple of 4, pad the input to the next multiple of 4 or use scalar fallback for the remaining elements."*

For production use, the skilled kernel should pad K to the next multiple of 4 (or add a scalar tail loop). The current implementation enforces the constraint at the launcher level — a pragmatic choice for benchmarking at power-of-2 sizes.
