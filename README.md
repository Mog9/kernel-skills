# kernel-skills

**kernel-skills** is an open source library of high quality skill files for AI coding agents working on compute kernels.

---

## What it is

kernel-skills is a curated collection of `SKILL.md` files. Each file is a structured engineering playbook that an AI coding agent can follow when writing, optimizing, debugging, or porting compute kernels.

This repository is not a runtime, framework, package manager, benchmark harness, or MCP server. It is a library of Markdown files. No installation required. No tooling required. Pick a skill, read it or paste it into your agent workflow, and get better kernel output.

---

## Why it exists

AI coding agents produce substantially worse kernel code when given vague prompts. They skip constraint gathering, choose incorrect tile strategies, ignore boundary conditions, make unsupported performance claims, and produce code that looks plausible but fails on real hardware.

Structured skill files change this. A well-authored skill forces the agent to:

- gather the right constraints before writing a single line of code
- choose the correct algorithm and memory strategy for the workload
- reason explicitly about correctness risks and edge cases
- avoid cargo-cult optimization and fake performance claims
- explain tradeoffs with technical precision
- know when a custom kernel is not the right answer

This repository exists to provide those skill files at expert quality, openly, for any agent and any workflow.

---

## Who it is for

This repository is for engineers who use AI coding agents to work on:

- CUDA kernel development
- Triton kernel development
- Quantized kernels (int8, fp8)
- High performance numerics and AI workloads
- Kernel optimization, debugging, and porting

It is also useful for engineers who want a technical reference for how to approach these problems systematically, independent of any agent.

---

## Repository structure

```
kernel-skills/
├── README.md
├── LICENSE
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── ROADMAP.md
├── CLAUDE.md
├── .gitignore
└── skills/
    └── cuda/
        ├── write-cuda-gemm-kernel/
        ├── write-cuda-reduction-kernel/
        ├── write-cuda-softmax-kernel/
        ├── write-cuda-layernorm-kernel/
        ├── optimize-global-memory-access/
        ├── optimize-shared-memory-tiling/
        ├── avoid-warp-divergence/
        ├── choose-launch-configuration/
        └── debug-cuda-kernel-correctness/
    └── triton/
        ├── write-triton-gemm-kernel/
        ├── write-triton-softmax-kernel/
        ├── write-triton-layernorm-kernel/
        ├── write-triton-attention-kernel/
        └── optimize-triton-block-parameters/
    └── patterns/
        ├── fuse-elementwise-ops/
        ├── write-numerically-stable-kernel/
        ├── handle-boundary-conditions/
        ├── choose-tile-size-and-work-partitioning/
        └── write-kernel-test-plan/
    └── quantization/
        ├── write-int8-quantized-kernel/
        ├── write-fp8-kernel/
        └── debug-quantized-kernel-accuracy/
```

More skills are being added. See [ROADMAP.md](ROADMAP.md) for what is coming next.

---

## Skills

### CUDA

| Skill | Description |
|---|---|
| [`write-cuda-gemm-kernel`](skills/cuda/write-cuda-gemm-kernel/SKILL.md) | Design and implement a tiled CUDA GEMM kernel — shared memory strategy, tensor core eligibility, accumulation precision, and when to use cuBLAS/CUTLASS instead |
| [`write-cuda-reduction-kernel`](skills/cuda/write-cuda-reduction-kernel/SKILL.md) | Write a correct parallel reduction with warp shuffle tree, multi-block strategy, and correct handling of partial tiles |
| [`write-cuda-softmax-kernel`](skills/cuda/write-cuda-softmax-kernel/SKILL.md) | Implement online or two-pass softmax with numerically stable max subtraction and correct warp-level reduction |
| [`write-cuda-layernorm-kernel`](skills/cuda/write-cuda-layernorm-kernel/SKILL.md) | Implement layer normalization with Welford online variance, fused mean/variance computation, and fp32 accumulation in fp16 kernels |
| [`optimize-global-memory-access`](skills/cuda/optimize-global-memory-access/SKILL.md) | Analyze and fix coalescing, alignment, and vectorized load/store patterns using Nsight Compute metrics |
| [`optimize-shared-memory-tiling`](skills/cuda/optimize-shared-memory-tiling/SKILL.md) | Apply shared memory tiling with bank conflict analysis, padding strategies, and double buffering |
| [`avoid-warp-divergence`](skills/cuda/avoid-warp-divergence/SKILL.md) | Classify avoidable vs unavoidable divergence, apply ballot/shuffle fast paths and stream compaction, estimate the real cost before restructuring |
| [`choose-launch-configuration`](skills/cuda/choose-launch-configuration/SKILL.md) | Select block size, grid size, and shared memory from occupancy analysis, register budget, and workload shape |
| [`debug-cuda-kernel-correctness`](skills/cuda/debug-cuda-kernel-correctness/SKILL.md) | Systematic workflow for isolating indexing bugs, race conditions, reduction errors, dtype issues, and out-of-bounds accesses in CUDA kernels |

### Triton

| Skill | Description |
|---|---|
| [`write-triton-gemm-kernel`](skills/triton/write-triton-gemm-kernel/SKILL.md) | Write a Triton GEMM kernel with correct block tiling, tl.dot accumulation, row/col-major loading, and when CUTLASS is preferable |
| [`write-triton-softmax-kernel`](skills/triton/write-triton-softmax-kernel/SKILL.md) | Implement numerically stable softmax in Triton with block size selection for the reduction axis and masking for variable sequence lengths |
| [`write-triton-layernorm-kernel`](skills/triton/write-triton-layernorm-kernel/SKILL.md) | Implement LayerNorm in Triton with Welford online variance, persistent kernel pattern, and backward pass accumulation strategy |
| [`write-triton-attention-kernel`](skills/triton/write-triton-attention-kernel/SKILL.md) | Implement Flash Attention in Triton — causal mask handling, kv-block loop structure, online softmax scaling, and fp16/bf16 accumulation decisions |
| [`optimize-triton-block-parameters`](skills/triton/optimize-triton-block-parameters/SKILL.md) | Select BLOCK_M/N/K, num_warps, and num_stages; reason about register pressure, occupancy, and autotuning config design |

### Patterns

| Skill | Description |
|---|---|
| [`fuse-elementwise-ops`](skills/patterns/fuse-elementwise-ops/SKILL.md) | Decide when and how to fuse elementwise operations — memory bandwidth arithmetic, producer-consumer fusion, and epilogue fusion patterns |
| [`write-numerically-stable-kernel`](skills/patterns/write-numerically-stable-kernel/SKILL.md) | Apply Kahan summation, log-sum-exp trick, compensated accumulation, and dtype selection for stable intermediate values |
| [`handle-boundary-conditions`](skills/patterns/handle-boundary-conditions/SKILL.md) | Handle partial tiles, misaligned sizes, and out-of-bounds accesses correctly — masked loads, predicated stores, and tail handling strategies |
| [`choose-tile-size-and-work-partitioning`](skills/patterns/choose-tile-size-and-work-partitioning/SKILL.md) | Reason about arithmetic intensity, shared memory budget, occupancy tradeoffs, and work partitioning for irregular shapes |
| [`write-kernel-test-plan`](skills/patterns/write-kernel-test-plan/SKILL.md) | Design a correctness and numerical test plan — reference comparison strategy, input shape sweep, dtype coverage, tolerance reasoning, and CI integration |

### Quantization

| Skill | Description |
|---|---|
| [`write-int8-quantized-kernel`](skills/quantization/write-int8-quantized-kernel/SKILL.md) | Implement INT8 quantized matrix operations — dp4a instruction, symmetric vs asymmetric quantization, INT32 accumulation, per-channel scale epilogue, cuBLAS vs CUTLASS vs custom decision |
| [`write-fp8-kernel`](skills/quantization/write-fp8-kernel/SKILL.md) | Design FP8 compute kernels for Hopper/Ada — E4M3/E5M2 format selection, satfinite conversion, delayed scaling, WGMMA on H100, and hipBLASLt on MI300X |
| [`debug-quantized-kernel-accuracy`](skills/quantization/debug-quantized-kernel-accuracy/SKILL.md) | Diagnose accuracy regressions in quantized kernels — scale validation, overflow detection, per-element error attribution, and calibration diagnostics |

---

## How to use a skill

1. Find the skill that matches your task in `skills/`.
2. Open the `SKILL.md` file and paste its full contents into your agent's context.
3. Ask the agent to perform the task.

The skill does not replace your prompt — it forces the agent to reason correctly before writing a single line of code.

### Example (Claude Code)

```
<paste contents of skills/cuda/write-cuda-reduction-kernel/SKILL.md>

Write a warp-shuffle reduction kernel for float32 inputs on an H100.
Input shape: [B=32, N=65536]. Output: [B] row-wise sums.
```

The skill works the same way with ChatGPT, Cursor, Gemini CLI, and any other agent that accepts context.

---

## Contributing

Contributions are welcome. Before opening a pull request, read [CONTRIBUTING.md](CONTRIBUTING.md).

The short version: open an issue first to propose the skill scope, follow the required 11-section `SKILL.md` template, meet the quality bar, and keep naming conventions consistent.

Low-quality, vague, or out-of-scope skill files will not be merged regardless of technical domain.

---

## Roadmap

More skills are being added across CUDA, Triton, quantization, and portability. Following the quality-first principle: each skill ships only when it is genuinely better than a generic prompt.

See [ROADMAP.md](ROADMAP.md) for the full plan.

---

## License

MIT. See `LICENSE`.
