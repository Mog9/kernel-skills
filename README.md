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
├── skills/
│   ├── cuda/
│   │   ├── write-cuda-gemm-kernel/
│   │   ├── write-cuda-reduction-kernel/
│   │   ├── write-cuda-softmax-kernel/
│   │   ├── write-cuda-layernorm-kernel/
│   │   ├── optimize-global-memory-access/
│   │   ├── optimize-shared-memory-tiling/
│   │   ├── avoid-warp-divergence/
│   │   ├── choose-launch-configuration/
│   │   └── debug-cuda-kernel-correctness/
│   ├── triton/
│   │   ├── write-triton-gemm-kernel/
│   │   ├── write-triton-softmax-kernel/
│   │   ├── write-triton-layernorm-kernel/
│   │   ├── write-triton-attention-kernel/
│   │   └── optimize-triton-block-parameters/
│   ├── patterns/
│   │   ├── fuse-elementwise-ops/
│   │   ├── write-numerically-stable-kernel/
│   │   ├── handle-boundary-conditions/
│   │   ├── choose-tile-size-and-work-partitioning/
│   │   └── write-kernel-test-plan/
│   ├── quantization/
│   │   ├── write-int8-quantized-kernel/
│   │   ├── write-fp8-kernel/
│   │   └── debug-quantized-kernel-accuracy/
│   └── portability/
│       ├── port-cuda-kernel-to-triton/
│       ├── port-cuda-kernel-to-hip/
│       └── write-backend-agnostic-kernel-plan/
└── examples/
    ├── how-to-use-with-claude-code.md
    ├── how-to-use-with-chatgpt.md
    ├── how-to-use-with-cursor.md
    └── how-to-use-with-gemini-cli.md
```

---

## Skills included in v0

### CUDA

| Skill | Description |
|---|---|
| `write-cuda-gemm-kernel` | Design and implement a CUDA GEMM kernel with correct tiling, shared memory usage, and layout awareness |
| `write-cuda-reduction-kernel` | Write a correct and efficient parallel reduction with warp-level primitives and multi-pass strategy |
| `write-cuda-softmax-kernel` | Implement online or two-pass softmax with numerically stable accumulation and correct reduction |
| `write-cuda-layernorm-kernel` | Implement layer normalization with fused mean/variance computation and correct accumulation |
| `optimize-global-memory-access` | Analyze and fix coalescing, alignment, and vectorized load/store patterns |
| `optimize-shared-memory-tiling` | Apply shared memory tiling with correct bank conflict analysis and synchronization |
| `avoid-warp-divergence` | Identify and eliminate control flow divergence within warps |
| `debug-cuda-kernel-correctness` | Systematic approach to isolating and fixing correctness bugs in CUDA kernels |

### Triton

| Skill | Description |
|---|---|
| `write-triton-gemm-kernel` | Write a Triton GEMM kernel with correct block tiling and memory access patterns |
| `write-triton-softmax-kernel` | Implement softmax in Triton with correct reduction and masking |
| `write-triton-layernorm-kernel` | Implement layer normalization in Triton with correct numerics |
| `write-triton-attention-kernel` | Implement attention (including flash attention style) in Triton |
| `optimize-triton-block-parameters` | Reason about and select block sizes, number of warps, and pipeline stages for Triton kernels |

### Patterns

| Skill | Description |
|---|---|
| `fuse-elementwise-ops` | Decide when and how to fuse elementwise operations into a single kernel |
| `write-numerically-stable-kernel` | Apply Kahan summation, online normalization, and log-sum-exp patterns correctly |
| `handle-boundary-conditions` | Handle partial tiles, misaligned sizes, and out-of-bounds accesses correctly |
| `write-kernel-test-plan` | Design a correctness and numerical test plan for a custom kernel |

### Quantization

| Skill | Description |
|---|---|
| `write-int8-quantized-kernel` | Implement int8 quantized operations with correct scale/zero-point handling |
| `debug-quantized-kernel-accuracy` | Diagnose accuracy regressions in quantized kernels |

### Portability

| Skill | Description |
|---|---|
| `port-cuda-kernel-to-triton` | Systematically translate a CUDA kernel to Triton, preserving correctness and intent |

---

## How to use a skill

### The basic workflow

1. Find the skill that matches your task in `skills/`.
2. Open the `SKILL.md` file and paste its contents into your agent's context (system prompt, conversation, or project instructions file).
3. Ask the agent to perform the task. The skill shapes how the agent reasons, what it asks before coding, and what it produces.

The skill does not replace your prompt. It augments the agent's reasoning process so the output is more correct, more explicit, and more technically grounded.

---

### Using with Claude Code

Paste the skill content directly into your conversation, or add it to your project's `CLAUDE.md` file so it applies to all relevant requests in that project.

```
# In your Claude Code session:
<paste contents of skills/cuda/write-cuda-reduction-kernel/SKILL.md>

Now write a warp-shuffle based reduction kernel for float32 inputs on an H100.
Input shape: [B, N] where B=32, N=65536. Output: [B] of per-row sums.
```

See `examples/how-to-use-with-claude-code.md` for a full walkthrough.

---

### Using with ChatGPT

Paste the skill content into the system prompt or at the top of the conversation. For Custom GPTs, add the skill content to the system instructions.

See `examples/how-to-use-with-chatgpt.md` for a full walkthrough.

---

### Using with Cursor

Add the skill content to your `.cursorrules` file or paste it into the Cursor chat before making your request.

See `examples/how-to-use-with-cursor.md` for a full walkthrough.

---

### Using with Gemini CLI

Paste the skill into your prompt, or add it to a `GEMINI.md` file in your project root for persistent context.

See `examples/how-to-use-with-gemini-cli.md` for a full walkthrough.

---

## Contributing

Contributions are welcome. Before opening a pull request, read `CONTRIBUTING.md`.

The short version: open an issue first to propose the skill scope, follow the required 11-section `SKILL.md` template, meet the quality bar, and keep naming conventions consistent.

Low-quality, vague, or out-of-scope skill files will not be merged regardless of technical domain.

---

## Roadmap

The v0 goal is simple: prove that well-authored skill files materially improve agent kernel output.

Future versions may add metadata, tagging, indexing, and eventually packaging — but only after v0 demonstrates clear value. See `ROADMAP.md` for details.

---

## License

MIT. See `LICENSE`.
