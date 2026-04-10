# Roadmap

## v0 — Skills only

The sole objective of v0 is to prove a single claim:

**Well-authored skill files materially improve the quality of kernel code produced by AI agents.**

This means the repository succeeds when engineers can pick up a skill, apply it to their agent workflow, and consistently get better output than they would from a generic prompt — more correct code, more explicit reasoning, fewer hidden assumptions, more honest performance analysis.

### What v0 includes

- 20 expert-grade SKILL.md files across five categories: CUDA, Triton, patterns, quantization, and portability
- Foundational repository documentation: README, CONTRIBUTING, ROADMAP, CODE_OF_CONDUCT, LICENSE
- Usage examples for Claude Code, ChatGPT, Cursor, and Gemini CLI
- No tooling, no packaging, no automation beyond standard repo hygiene

### What v0 does not include

- CLI tools
- MCP server integrations
- Benchmark runners or execution frameworks
- Package managers or installable distributions
- Metadata systems or tagging infrastructure

These are not planned for v0. They may be explored later, but only if the core skill library has demonstrated clear value first.

---

## v1 — Metadata, indexing, curation (speculative)

If v0 proves useful, v1 may explore:

- Structured metadata in skill files: tags for architecture target (A100, H100, RDNA), kernel type, backend, precision
- A browsable index of skills filterable by tag
- Quality signals: community ratings, known issues, last-reviewed date
- Additional skills in under-covered areas based on community feedback
- Broader coverage of quantization (fp8, mixed-precision workflows)
- Additional portability skills (CUDA to ROCm/HIP, cross-platform patterns)

None of this is committed. It depends entirely on whether v0 generates genuine demand.

---

## Future possibilities (speculative, not planned)

After demonstrated traction:

- Packaging: npm or pip installable skill sets for agent project scaffolding
- MCP integration: skills surfaced as MCP resources for agent-native discovery
- Agent-specific skill variants: versions of skills tuned to how specific agents (Claude, GPT-4, Gemini) respond best to structured prompts
- Automated testing: tooling that evaluates whether agent output with a skill is better than output without it

These will not be built speculatively. If v0 works, the right next steps will be obvious from how people actually use it.

---

## What this project will not become

Regardless of traction, this project will not become:

- A kernel benchmark suite
- A code execution platform
- A tutorials site
- A catch-all GPU programming reference
- A collection of vague prompt templates

The focus is narrow by design. A repository that does one thing well is more valuable than one that tries to do everything.
