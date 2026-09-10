# Understanding Agents

This repository contains small LangGraph examples that demonstrate common agent patterns, workflows, and state definitions.

## Included examples

### HumanInTheLoop

A LinkedIn post generator that pauses for human approval after each draft. The workflow uses a writer LLM, a LangGraph interrupt for review, and MemorySaver so the run can resume after feedback.

### Iterative-tools

A LinkedIn post generator that writes, reviews, and rewrites until the content is approved. It can use Tavily search for up-to-date facts and loops through writer, tool/extract, and reviewer steps.

### parallel-reducers

A safety analyzer that scores a script for toxicity, copyright risk, and cultural sensitivity. Three LangGraph nodes run in parallel and merge their results into a single shared safety score.

### seq_wrkflow

A sequential LangGraph workflow that cleans raw text, turns it into a script, and produces a final polished output.

### states

Example patterns for defining LangGraph-style shared state in Python using `TypedDict`, `Pydantic BaseModel`, and `dataclass`.

## Repository structure

- `src/` — example workflows and supporting code
- `main.py` — entry point for running the examples
- `requirements.txt` — Python dependencies
- `README.md` — project overview
