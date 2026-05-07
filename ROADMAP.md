# Roadmap

## v0 — Skills only

The sole objective of v0 is to prove a single claim:

**Well-authored skill files materially improve the quality of kernel code produced by AI agents.**

This means the repository succeeds when engineers can pick up a skill, apply it to their agent workflow, and consistently get better output than they would from a generic prompt — more correct code, more explicit reasoning, fewer hidden assumptions, more honest performance analysis.

### What v0 includes

- 35 expert-grade SKILL.md files across six categories: CUDA, Triton, patterns, quantization, portability, and inference
- Foundational repository documentation: README, CONTRIBUTING, ROADMAP, CODE_OF_CONDUCT, LICENSE
- Usage examples for Claude Code, ChatGPT, Cursor, and Gemini CLI
- An npm package (`@krxgu/kernel-skills`) with a CLI, programmatic API, and JSON-Schema-validated `skill.json` metadata — so skills can be searched, fetched, and bundled into agent contexts programmatically

### Shipped milestones

- **v0.1.0** — initial public release (25 skills): CUDA, Triton, patterns, quantization, portability
- **v0.2.0** — inference coverage: 10 new skills for the LLM serving hot path (RMSNorm, fused add+norm, SwiGLU, RoPE, sampling, KV cache, dequant, prefill-vs-decode, vLLM and TensorRT integration plans)

### What v0 does not include

- MCP server integrations
- Benchmark runners or execution frameworks
- Code execution or kernel compilation
- Agent-specific variants of skills

These are not planned for v0. They may be explored later, but only if the core skill library demonstrates clear value first.

---

## What's next for v0

The skill library and the npm package are the foundation. Future v0.x releases focus on filling remaining coverage gaps and producing measurable evidence that the skills actually improve agent output:

- More inference skills: speculative decoding, structured output / constrained sampling, MoE routing, FlashInfer-style kernels
- More quantization skills: AWQ-fused-GEMM, marlin, FP8 inference flows
- More patterns: persistent kernels, Hopper-specific TMA / WGMMA patterns
- More portability skills: CUDA → Metal, CUDA → SYCL
- The `proof/` directory: per-skill before/after evidence (correctness heatmaps, bandwidth charts) so the impact of each skill is visible, not asserted

---

## v1 — Speculative

Considered only if v0 demonstrates clear value:

- A browsable index of skills filterable by tag, hardware target, difficulty
- Quality signals: community ratings, known issues, last-reviewed date
- Agent-specific skill variants: versions tuned to how specific agents (Claude, GPT, Gemini) respond best to structured prompts
- MCP integration: skills surfaced as MCP resources for agent-native discovery
- Automated evaluation: tooling that measures whether agent output with a skill is better than output without it

None of this is committed. It depends entirely on whether the v0 library generates genuine demand.

---

## What this project will not become

Regardless of traction, this project will not become:

- A kernel benchmark suite
- A code execution platform
- A tutorials site
- A catch-all GPU programming reference
- A collection of vague prompt templates

The focus is narrow by design. A repository that does one thing well is more valuable than one that tries to do everything.
