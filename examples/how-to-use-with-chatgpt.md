# How to Use kernel-skills with ChatGPT

ChatGPT (including GPT-4o) works well with `SKILL.md` files when you paste them directly into your conversation. The skill acts as a detailed system-level instruction that shapes how ChatGPT reasons about the kernel task.

---

## Basic usage pattern

1. Open the relevant `SKILL.md` file in this repository.
2. Copy the full contents of the file.
3. In a new ChatGPT conversation, paste the skill content at the start of your message, followed by your task description.

### Example

```
I want you to follow this skill when writing the kernel:

---
<paste contents of skills/cuda/write-cuda-gemm-kernel/SKILL.md>
---

Now write a CUDA GEMM kernel for:
- A: float16, shape [M=2048, K=4096], row-major
- B: float16, shape [K=4096, N=2048], column-major
- C: float16, shape [M=2048, N=2048], row-major
- Target: sm_80 (A100)
- Do not use cuBLAS. Implement a tiled GEMM with shared memory.
```

ChatGPT will follow the required reasoning process, gather missing information from the inputs section, and apply the kernel design rules before writing code.

---

## Using the Custom GPT system prompt (advanced)

If you use ChatGPT with custom instructions (Settings → Personalization → Custom instructions), you can paste the skill content into the "What would you like ChatGPT to know about you?" or "How would you like ChatGPT to respond?" fields.

This is useful if you are routinely working on a specific class of kernel and want the skill applied automatically without pasting it each time.

**Limitation**: custom instructions are limited in length. Use this only for a single short skill or a condensed version of the rules section.

---

## Breaking the task into steps

For complex kernel writing tasks, prompting ChatGPT to follow the skill step-by-step produces better results than asking for the full implementation in one shot.

### Example: step-by-step reasoning

After pasting the skill:

```
Let's follow the required reasoning process one step at a time.

Step 1: Gather the inputs. Ask me any clarifying questions before writing any code.
```

Wait for ChatGPT to ask its questions, answer them, then continue:

```
Step 2: Apply the kernel design approach based on my answers. Describe your strategy without writing code yet.
```

Then:

```
Step 3: Write the kernel following the strategy you described. Apply the review checklist before finishing.
```

---

## Using ChatGPT for debugging with a skill

For debugging sessions, paste the debug skill and provide the failing kernel and test case:

```
Follow this debugging skill:

---
<paste contents of skills/cuda/debug-cuda-kernel-correctness/SKILL.md>
---

My kernel produces NaN outputs for input size 128x127 but correct results for 128x128.
Here is the kernel:

[paste kernel]

Classify the bug using the error classification from the skill, then find the root cause.
```

---

## Tips for best results

- Ask ChatGPT to "follow the review checklist before finishing" at the end of a kernel writing task. This prevents common omissions (missing boundary handling, unsupported performance claims).
- If ChatGPT skips the "Inputs to gather" step and goes straight to code, remind it: "The skill requires you to gather inputs first before generating code."
- For large skills, pasting the full file is better than summarizing it. The specific rules and failure modes are what make the skill valuable — paraphrasing them loses precision.
- Use GPT-4o rather than GPT-3.5 for kernel tasks. The quality gap is significant for technically demanding code generation.
