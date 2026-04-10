# How to Use kernel-skills with Gemini CLI

Gemini CLI is Google's terminal-based AI coding agent. It supports context injection via `GEMINI.md` files, direct file references, and inline prompting — all of which work well with kernel-skills `SKILL.md` files.

---

## Basic usage pattern: paste into the prompt

The most direct approach: copy the `SKILL.md` content and include it in your Gemini CLI prompt.

### Example

```bash
gemini "$(cat skills/cuda/write-cuda-reduction-kernel/SKILL.md)

Using the skill above, write a parallel warp-shuffle reduction kernel:
- Input: float32, shape [N=65536]
- Operation: sum reduction to a scalar
- Target: A100 (sm_80)
- Use warp shuffle intrinsics for the final warp reduction
- Provide a host-side launcher"
```

The `$(cat ...)` shell substitution inlines the skill content directly into the prompt string. Gemini CLI will respect the structured reasoning process in the skill.

---

## Method 2: Use a `GEMINI.md` file for persistent context

Gemini CLI reads a `GEMINI.md` file from the current working directory (similar to Claude Code's `CLAUDE.md`). You can place skill content in `GEMINI.md` to apply it to all interactions in a project session.

### Setup

```bash
# In your kernel project root:
cat skills/cuda/write-cuda-gemm-kernel/SKILL.md > GEMINI.md
```

Or combine multiple skills:

```bash
cat skills/cuda/write-cuda-gemm-kernel/SKILL.md \
    skills/cuda/optimize-shared-memory-tiling/SKILL.md \
    > GEMINI.md
```

Then launch Gemini CLI from that directory:

```bash
gemini
```

All interactions in that session will follow the skill instructions without needing to re-paste the content.

---

## Method 3: Reference skill files directly

If kernel-skills is cloned inside your working directory or as a submodule, you can reference files directly in your prompt:

```bash
gemini "@skills/cuda/debug-cuda-kernel-correctness/SKILL.md

My kernel produces wrong results for inputs with N=2047. Here is the kernel:
$(cat src/my_reduction.cu)

Classify the bug using the skill and identify the root cause."
```

The `@` file reference syntax tells Gemini CLI to read the file and include it in context.

---

## Method 4: Multi-turn sessions with `--interactive`

For iterative kernel development, use Gemini CLI's interactive mode and include the skill at the start of the session:

```bash
gemini --interactive
```

In the first message:

```
I want you to follow this kernel engineering skill throughout our conversation:

[paste skill content]

Acknowledge the skill and tell me what inputs you need before writing any code.
```

Then continue the session with follow-up messages as you refine the kernel.

---

## Example: optimization workflow

```bash
# Step 1: Write a first-draft kernel
gemini "$(cat skills/cuda/write-cuda-layernorm-kernel/SKILL.md)
Write a float16 LayerNorm kernel for shape [batch=64, hidden=4096], target sm_80."

# The output kernel is saved as layernorm_v1.cu

# Step 2: Optimize it
gemini "$(cat skills/cuda/optimize-shared-memory-tiling/SKILL.md)
Kernel to optimize:
$(cat layernorm_v1.cu)
Apply the shared memory tiling skill to identify memory access bottlenecks and propose improvements."
```

---

## Tips for best results

- For long skills, the `$(cat file)` substitution is more reliable than manual copy-paste since it avoids formatting loss.
- Gemini CLI works well when the skill's "Inputs the agent should gather first" list is pre-filled. Provide all inputs (shapes, dtypes, hardware target) in the first message to avoid back-and-forth.
- When asking Gemini to follow the review checklist: explicitly paste the checklist section at the end of your prompt and say "check each item and confirm compliance before finishing."
- For multi-skill sessions, ordering matters: put the "write" skill before the "test plan" skill, or ask for one deliverable at a time per skill rather than requesting everything simultaneously.
- Gemini 2.0 Pro and Ultra provide substantially better code quality than earlier versions for GPU kernel tasks. Use the largest available model for kernel engineering work.
