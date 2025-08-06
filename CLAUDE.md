# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an EmbeddingService project for SlimRAG - a Python-based service that processes various file types and generates embeddings for vector search and retrieval applications.

## Technology Stack

- **Python**: >=3.12
- **Package Manager**: uv (modern Python package manager)
- **CLI Framework**: fire (for command-line interface)
- **Document Processing**: markitdown, MinerU
- **Embedding Models**: bge-m3, Qwen3-Embedding, Chinese-CLIP
- **Data Storage**: DuckDB (metadata), Meilisearch (search)

## Data Processing Pipeline

The service handles different file types with specific processing chains:

- `.md` files → directly processed with bge-m3
- `.pdf`, `.doc`, `.docx` → MinerU extraction → bge-m3
- `.html` files → markitdown conversion → bge-m3
- `.webp` images → converted to `.jpg` → CLIP
- `.jpg`, `.png` images → directly processed with CLIP
- `.arw` (RAW) → converted to `.jpg` + demosaic → CLIP
- `.mp4` videos → keyframe extraction → CLIP

## Key Dependencies

- **fire**: CLI framework for creating command-line interfaces
- **markitdown**: HTML to Markdown conversion
- **mineru[core]**: Document processing and extraction (>=2.1.10)

## Development Commands

Since this is a Python project using uv, common development commands include:

```bash
# Install dependencies
uv sync

# Run the service (implementation specific)
uv run python -m embeddingservice

# Add new dependencies
uv add <package_name>

# Run tests (when implemented)
uv run pytest
```

## Architecture Notes

This appears to be a multi-modal embedding service that:

1. Processes various document formats (text, PDFs, Office docs)
2. Handles different image formats (including RAW camera files)
3. Extracts video keyframes
4. Uses multiple embedding models for different content types
5. Supports clustering with CLIP + scikit-learn
6. Stores metadata in DuckDB and provides search via Meilisearch

The project is in early stages with minimal implementation files, focusing on the data processing pipeline architecture.
