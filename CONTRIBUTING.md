# Contributing to kernel-skills

Thank you for your interest in contributing. This document explains how to propose, author, and submit skill files that meet the quality standards of this repository.

Read this document fully before opening a pull request. PRs that do not follow these guidelines will not be merged.

---

## What this repository accepts

kernel-skills accepts:

- New `SKILL.md` files for kernel engineering topics within the defined scope
- Improvements to existing skill files that increase technical accuracy, depth, or clarity
- Corrections to factually incorrect claims in existing skills

kernel-skills does not accept:

- Tooling, CLI utilities, or execution frameworks
- MCP server integrations
- Benchmark harnesses or performance measurement infrastructure
- Tutorial-style content or beginner introductions
- Skills for out-of-scope domains (see below)
- Vague, shallow, or low-quality skill files regardless of topic

---

## Out of scope

The following are explicitly out of scope for v0:

- Linux kernel development (device drivers, OS internals)
- Graphics shader pipelines as a primary focus
- Full compiler toolchain design
- Packaging systems or distribution frameworks
- Benchmark runners or execution tooling
- Skills that duplicate or trivially restate existing skills

---

## Step 1: Open an issue before writing

Before writing a skill, open a GitHub issue with:

- The proposed skill name (following naming conventions below)
- The exact scope: what task does this skill help an agent perform?
- Why existing skills do not cover this need
- The target backend or domain (CUDA, Triton, patterns, quantization, portability)
- A one-paragraph description of what makes this skill technically non-trivial

This prevents wasted effort on skills that are out of scope, redundant, or too broad to be useful.

Maintainers will respond to confirm scope or suggest adjustments before you write anything.

---

## Step 2: Follow the required SKILL.md structure

Every skill file must follow this exact 11-section template. Do not add sections arbitrarily. Do not omit sections.

```markdown
# Skill: <Skill Name>

## Purpose
State exactly what the skill helps the agent do.

## Use this when
List the situations where this skill is the right tool.

## Do not use this when
List situations where this skill should not be used.

## Inputs the agent should gather first
List all required information that must be clarified before generating code.

## Required reasoning process
Provide a numbered reasoning workflow that the agent must follow.

## Kernel design rules
State strict implementation and optimization principles.

## Correctness requirements
State what must be true for the output to be considered correct.

## Performance requirements
State what the agent must reason about to justify performance decisions.

## Output format
Specify the required structure of the final answer the agent should produce.

## Common failure modes
List realistic bugs and design mistakes.

## Review checklist
Provide a final quality checklist the agent should self-apply before finishing.
```

If a skill genuinely requires one additional section beyond this template, document the reason clearly in your PR description. Do not add sections without justification.

---

## Step 3: Meet the quality bar

Every skill submitted to this repository must meet the following quality standard.

### Required

- **Specific scope**: The skill covers a concrete, well-defined task. It is not a general introduction to GPU programming.
- **Expert-grade content**: The skill should be useful to a working kernel engineer, not just a student encountering the topic for the first time.
- **Actionable reasoning process**: The "Required reasoning process" section must be a numbered workflow the agent can actually follow, not a list of aspirations.
- **Realistic failure modes**: The "Common failure modes" section must list real bugs and design mistakes that occur in practice, not theoretical concerns.
- **Honest performance guidance**: Do not assert that a custom kernel will outperform vendor libraries without evidence. Do not promise speedups. Use language like "likely reduces memory traffic because..." and "this tradeoff should be benchmarked against...".
- **Correctness-first framing**: Correctness must be addressed before performance optimization in every skill that involves code generation.
- **Architecture awareness**: Skills must not give generic GPU advice that ignores real hardware constraints like register pressure, occupancy, shared memory limits, and warp size.

### Not allowed

- Vague optimization tips ("make sure memory access is coalesced" with no specifics)
- Generic advice that adds no value over asking the agent without a skill
- Fake authority ("this will be 3x faster than...")
- Beginner tutorial tone
- Boilerplate filler text
- Sections that are present but empty or trivial
- Cargo-cult optimization patterns without explanation of why they apply

---

## Naming conventions

Skill folder names must be:

- Lowercase
- Hyphen-separated
- Descriptive and action-oriented where possible
- Immediately clear about what the skill does

Good examples:
- `write-cuda-gemm-kernel`
- `optimize-global-memory-access`
- `debug-quantized-kernel-accuracy`
- `port-cuda-kernel-to-triton`
- `handle-boundary-conditions`

Bad examples:
- `gemm` (too vague)
- `cuda-optimization` (too broad)
- `better-kernels` (meaningless)
- `WriteGemmKernel` (wrong case and format)
- `fast-gemm-tips` (vague, tutorial tone)

---

## Directory placement

Place new skills in the correct subdirectory:

| Domain | Directory |
|---|---|
| CUDA kernels and CUDA-specific optimization | `skills/cuda/` |
| Triton kernels | `skills/triton/` |
| General kernel patterns (fusion, numerics, tiling, testing) | `skills/patterns/` |
| Quantized kernels (int8, fp8, etc.) | `skills/quantization/` |
| Cross-backend porting and portability planning | `skills/portability/` |

If a skill genuinely spans two domains, place it in the most specific applicable directory and note the cross-domain relevance in the skill's "Use this when" section.

---

## Pull request process

1. Open an issue and get scope confirmation before writing.
2. Create a branch named `skill/<skill-folder-name>` (e.g., `skill/write-cuda-attention-kernel`).
3. Add the skill directory and `SKILL.md` file.
4. Verify your skill passes the self-review checklist below before submitting.
5. Open a pull request with a clear description of what the skill covers and why it meets the quality bar.
6. Respond to reviewer feedback. PRs that go stale without response will be closed.

---

## Self-review checklist before submitting

Before opening a pull request, verify:

- [ ] Is the scope specific and concrete?
- [ ] Does the skill explain exactly when to use it?
- [ ] Does the skill explain when not to use it?
- [ ] Does it require the agent to gather missing constraints before coding?
- [ ] Is the reasoning process a numbered, followable workflow?
- [ ] Does it discuss correctness risks with specifics?
- [ ] Does it discuss performance tradeoffs honestly?
- [ ] Does it avoid vague optimization claims?
- [ ] Does it avoid fake performance numbers or unsupported speed claims?
- [ ] Would an experienced kernel engineer find this useful?
- [ ] Is every section substantive (no filler, no placeholder text)?
- [ ] Does the naming follow conventions?
- [ ] Is the file placed in the correct directory?

If the answer to any of these is no, fix the skill before submitting.

---

## Improving existing skills

To improve an existing skill:

- Open a PR with the specific change and a clear explanation of what was wrong or incomplete
- If the change is factually significant, cite a source or explain the technical reasoning
- Do not rewrite a skill's structure without first opening an issue

---

## Questions

Open a GitHub issue with the label `question` if you are unsure whether a skill idea fits scope or how to approach a section.
