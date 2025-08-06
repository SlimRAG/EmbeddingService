# Draft

## Data pipeline

- `.md` => bge-m3
- `.pdf`, `.doc`, `.docx` => MinerU => bge-m3
- `.html` => markitdown => bge-m3
- `.webp` => `.jpg` => CLIP
- `.jpg`, `.png` => CLIP
- `.arw` => `.jpg` + demosaic => CLIP
- `.mp4` => Extract keyframes => CLIP

## Models

- [bge-m3](https://huggingface.co/BAAI/bge-m3)
- [Qwen3-Embedding](https://github.com/QwenLM/Qwen3-Embedding)
- [Chinese-CLIP](https://github.com/OFA-Sys/Chinese-CLIP)

## Clustering

CLIP + scikit-learn

## File metadata

DuckDB

## Recall

Meilisearch
