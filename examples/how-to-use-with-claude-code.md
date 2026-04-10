# How to Use kernel-skills with Claude Code

Claude Code is a terminal-based AI coding agent from Anthropic. You can paste a `SKILL.md` file directly into your request to guide Claude's reasoning process when writing, debugging, or optimizing compute kernels.

---

## Basic usage pattern

1. Open the `SKILL.md` file for the task you are working on.
2. Copy its contents.
3. Start a Claude Code session and paste the skill at the beginning of your request, followed by your specific task.

### Example

```
<paste contents of skills/cuda/write-cuda-reduction-kernel/SKILL.md>

Using the skill above, write a warp-shuffle reduction kernel for the following:
- Input: float32, shape [1024, 2048]
- Operation: sum reduction along the last dimension
- Output: float32, shape [1024]
- Target: A100 (sm_80)
- Expected output dtype: float32
```

Claude Code will use the skill's reasoning process and design constraints to produce a more carefully considered kernel than it would from a bare prompt.

---

## Using skills in ongoing sessions

For sessions where you are iterating on a kernel across multiple messages, include the skill in your first message. You do not need to repeat it in follow-up messages — Claude Code retains it in context.

### Example: iterative debugging session

```
<paste contents of skills/cuda/debug-cuda-kernel-correctness/SKILL.md>

I have a softmax kernel that produces wrong results for input width 63 but correct results for width 64. Here is the kernel:

[paste kernel code]
```

Then follow up with specific questions:

```
Using the classification from the skill: is this a boundary condition bug or an indexing bug?
```

---

## Combining multiple skills

For complex tasks it can be useful to provide two skills — for example, a "write" skill and a "correctness" skill together.

### Example: write and verify

```
<paste contents of skills/triton/write-triton-attention-kernel/SKILL.md>
<paste contents of skills/patterns/write-kernel-test-plan/SKILL.md>

Write a causal attention kernel in Triton for:
- Q, K, V: float16, shape [batch=4, heads=8, seq=2048, head_dim=64]
- Causal mask: yes
- Target: H100 (sm_90)

After writing the kernel, also produce a test plan using the test plan skill.
```

---

## Tips for best results

- Provide all of the "Inputs the agent should gather first" fields listed in the skill. If you skip them, Claude Code will ask for them before writing code.
- Specify hardware targets explicitly (e.g., "A100 sm_80", "H100 sm_90a"). Skills use this to make concrete tiling and type decisions.
- If Claude Code produces code that violates a rule in the skill (e.g., hardcodes a tile size without justification), quote the specific rule back to it and ask for a revision.

---

## Referencing skills from CLAUDE.md

If you are working in a repository that has a `CLAUDE.md` file (like this one), you can instruct Claude Code to reference specific skills by path:

```
Read skills/cuda/optimize-shared-memory-tiling/SKILL.md and apply it to the kernel in src/attention.cu
```

Claude Code will read the file automatically from the workspace.
