# TEACHME

This directory contains interactive lesson documents designed to be loaded into an LLM (Claude, GPT-4, or similar) to teach a user the fVDB API interactively.

## How it works

Each lesson is a self-contained markdown file that serves as both a system prompt and a curriculum. Load it by pasting the file contents as a system prompt, or attaching it as a document in your LLM session. The LLM will act as an interactive instructor, quizzing the student and adapting to their responses.

Each lesson includes:
- Teacher instructions (persona, pacing, scope)
- Module-by-module curriculum with embedded concepts and code examples
- Quiz questions and answer key
- Exercises with progressive difficulty
- A capstone project

An accompanying cheat sheet provides a quick API reference for use while coding.

## Available lessons

| Lesson | Cheat Sheet | Covers |
|---|---|---|
| [fvdb_core_lesson.md](fvdb_core_lesson.md) | [fvdb_core_cheatsheet.md](fvdb_core_cheatsheet.md) | `GridBatch`, `JaggedTensor`, sampling/splatting, sparse convolution, grid hierarchy, U-Net backbone |

## Not covered

- Gaussian Splatting (`fvdb.GaussianSplat3d`, `fvdb.gaussian_splatting`) — not yet included
- Visualization (`fvdb.viz`) — not yet included
- Volume rendering and ray marching — not yet included

