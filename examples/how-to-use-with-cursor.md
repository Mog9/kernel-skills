# How to Use kernel-skills with Cursor

Cursor is an AI-powered code editor built on VS Code. Skills from this repository integrate naturally into Cursor via the chat panel, inline code generation, and the `.cursorrules` system.

---

## Method 1: Paste the skill into the Cursor chat panel

The simplest approach: open the Cursor chat (Cmd+L on Mac, Ctrl+L on Windows), paste the skill content at the start of your message, then describe your task.

### Example

```
Apply the following skill to write the kernel I need:

---
<paste contents of skills/cuda/write-cuda-softmax-kernel/SKILL.md>
---

Write a fused softmax kernel for:
- Input: float16, shape [batch=32, seq=4096]
- Output: float16, same shape
- Operation: online softmax (numerically stable, single pass)
- Target: A100 (sm_80)
```

Cursor's AI will use the skill's reasoning process to guide its implementation rather than producing a generic softmax.

---

## Method 2: Add the skill to `.cursorrules`

Cursor supports a project-level `.cursorrules` file (placed at the root of your workspace) that is automatically included in every AI interaction in that project.

To apply a skill globally for a kernel project:

1. Create or open `.cursorrules` in your project root.
2. Paste the skill content into the file (or a concise summary of the rules section).
3. Cursor will apply these rules in every chat and inline edit session within the project.

### Example `.cursorrules` for a CUDA project

```
When writing or editing CUDA kernels, follow these rules:

[paste the ## Kernel design rules section from the relevant SKILL.md files]
[paste the ## Correctness requirements section]
[paste the ## Review checklist section]
```

**Note**: `.cursorrules` has a size limit. If you are using many skills, include only the rules sections rather than the full skill files. Place full skill pastes in the chat when needed for specific tasks.

---

## Method 3: Reference skills using `@file` in the chat

If the kernel-skills repository is open in your Cursor workspace (or if the skills directory is inside your project), you can reference a skill file directly in the chat using the `@` file reference:

```
@skills/cuda/optimize-shared-memory-tiling/SKILL.md

Apply the shared memory tiling skill to refactor this GEMM kernel:

@src/gemm.cu
```

Cursor will read both files and apply the skill to the existing kernel. This is the most efficient workflow when working in a repository that already has kernel-skills cloned as a dependency or submodule.

---

## Method 4: Inline generation with Cmd+K

For smaller tasks — adding a specific optimization or fixing a boundary condition — use Cursor's inline generation (Cmd+K on Mac):

1. Select the code region to improve.
2. Press Cmd+K.
3. Type your instruction referencing the relevant skill rule. Example:

```
Following the shared memory tiling skill: add double-buffered shared memory tile loading to this kernel to overlap data transfer and computation.
```

Inline generation works best with specific, single-focus instructions. For complete kernel design from scratch, use the chat panel with a full skill paste.

---

## Tips for best results

- When using `@file` references for skill files, Cursor will include the full skill content in its context. This is cleaner than pasting.
- For multi-file kernel projects (kernel header, kernel implementation, test file), reference all three files in the chat along with the skill to give the model complete context.
- After receiving a kernel, ask Cursor to "self-apply the review checklist from the skill before finishing" to trigger a self-audit pass.
- Cursor's inline edit (Cmd+K) is well suited for targeted optimization steps (e.g., "apply the warp divergence avoidance strategy from the skill to this loop"). Use it iteratively rather than asking for a complete rewrite in one shot.
