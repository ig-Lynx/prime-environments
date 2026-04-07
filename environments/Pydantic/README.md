# Pydantic

## Overview

- **Environment ID**: `Pydantic`
- **Description**: Minimal scaffold environment for Pydantic-based tasks. Currently returns a dummy dataset with passthrough parser and zero-reward rubric. Reserved for future Pydantic-specific task implementations.
- **Tags**: pydantic, json, scaffold, single-turn

## Status

- Installable scaffold environment (passes CI checks)
- Not yet a runnable evaluation environment with real tasks

## Quickstart

```bash
uv run vf-eval Pydantic
```

## Related environment

If you intended to run the existing Pydantic-based environment in this repo, use:

- `pydantic-adherence` (see `../pydantic_adherence/`)

```bash
uv run vf-eval pydantic-adherence
```
