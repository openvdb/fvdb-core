# TEACHME

This directory contains interactive lesson documents designed to be loaded by an LLM coding agent (Claude Code, Cursor, or similar) to teach a user the fVDB API interactively.

## Getting started

The easiest way to use these lessons is with [Claude Code](https://docs.anthropic.com/en/docs/claude-code) (CLI or IDE extension) from the root of this repository. Claude Code can read the lesson files *and* the fvdb source code, so it can verify API details and help you debug exercises in real time.

**Example prompt to get started:**

```
Read docs/TEACHME and teach me how to use fvdb-core.
```

You can also give the LLM context about your background so it can tailor the lesson:

```
Read docs/TEACHME and teach me how to use fvdb-core. I have basic knowledge
of computer graphics and introductory experience with deep learning / PyTorch.
```

Keep `fvdb_core_cheatsheet.md` open in your editor while working through exercises — it's a quick API reference for the concepts covered in the lesson.

## How it works

Each lesson is a self-contained markdown file that serves as both a curriculum and instructor prompt. The LLM acts as an interactive instructor, teaching concepts module by module, quizzing the student, and adapting to their responses.

Each lesson includes:
- Teacher instructions (persona, pacing, scope)
- Module-by-module curriculum with embedded concepts and code examples
- Quiz questions and answer key
- Exercises with progressive difficulty
- A capstone project

## Available lessons

| Lesson | Cheat Sheet | Covers |
|---|---|---|
| [fvdb_core_lesson.md](fvdb_core_lesson.md) | [fvdb_core_cheatsheet.md](fvdb_core_cheatsheet.md) | `GridBatch`, `JaggedTensor`, sampling/splatting, sparse convolution, grid hierarchy, U-Net backbone |

## Not covered

- Gaussian Splatting (`fvdb.GaussianSplat3d`, `fvdb.gaussian_splatting`) — not yet included
- Visualization (`fvdb.viz`) — not yet included
- Volume rendering and ray marching — not yet included

