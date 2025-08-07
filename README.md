# EmbeddingService

A multi-modal embedding service for SlimRAG that processes various file types and generates embeddings for vector search and retrieval applications.

## Features

- **Multi-format Support**: Process documents, images, and videos
- **Multiple Embedding Models**: bge-m3, Qwen3-Embedding, Chinese-CLIP
- **Smart Processing Pipeline**: Different handling for each file type
- **Metadata Storage**: DuckDB for structured metadata
- **Search Integration**: Meilisearch for fast retrieval
- **CLI Interface**: Easy-to-use command-line tools

## Supported File Types

### Documents

- **Markdown (.md)**: Direct processing with bge-m3
- **PDF (.pdf)**: MinerU extraction bge-m3
- **Word (.doc, .docx)**: MinerU extraction bge-m3
- **HTML (.html)**: markitdown conversion bge-m3

### Images

- **JPEG (.jpg)**: Direct processing with CLIP
- **PNG (.png)**: Direct processing with CLIP
- **WebP (.webp)**: Convert to JPG CLIP
- **RAW (.arw)**: Convert to JPG + demosaic CLIP

### Videos

- **MP4 (.mp4)**: Keyframe extraction CLIP

## Installation

### Prerequisites

- Python 3.12 or higher
- [uv](https://docs.astral.sh/uv/) package manager

### Setup

1. Clone the repository:

```bash
git clone <repository-url>
cd EmebeddingService
```

2. Install dependencies:

```bash
uv sync
```

## Usage

### Command Line Interface

The service provides a comprehensive CLI through the `fire` framework:

```bash
# Process a single file
uv run python -m embeddingservice process_file "path/to/file.pdf"

# Process all files in a directory
uv run python -m embeddingservice process_directory "path/to/directory"

# Search for similar documents
uv run python -m embeddingservice search "your query here"

# Get file metadata
uv run python -m embeddingservice get_metadata "path/to/file.pdf"

# List all processed files
uv run python -m embeddingservice list_files

# Filter by content type
uv run python -m embeddingservice list_files --content_type "document"

# Get service status
uv run python -m embeddingservice status

# Configuration management
uv run python -m embeddingservice config show
uv run python -m embeddingservice config create --config_path "config.json"
```

### Configuration

The service can be configured through environment variables or a configuration file:

```bash
# Show current configuration
uv run python -m embeddingservice config show

# Create configuration file
uv run python -m embeddingservice config create --config_path "config.json"
```

## Architecture

### Processing Pipeline

1. **File Type Detection**: Automatic identification of file format
2. **Content Extraction**: Format-specific extraction (MinerU, markitdown, etc.)
3. **Embedding Generation**: Model-specific vector generation
4. **Metadata Storage**: Structured data in DuckDB
5. **Search Indexing**: Meilisearch for fast retrieval

### Core Components

- **CLI Interface**: Command-line interaction via `fire`
- **Processors**: Format-specific content extraction
- **Embedding Models**: Multiple model support (bge-m3, CLIP, etc.)
- **Storage Layer**: DuckDB for metadata, Meilisearch for search
- **Configuration**: Flexible configuration system

## Dependencies

### Core Dependencies

- **fire**: CLI framework
- **markitdown**: HTML to Markdown conversion
- **mineru[core]**: Document processing (e2.1.10)
- **torch**: Deep learning framework
- **transformers**: Model loading and inference
- **pillow**: Image processing
- **duckdb**: Metadata storage
- **meilisearch**: Search engine
- **opencv-python**: Video and image processing

### Development Dependencies

- **pytest**: Testing framework
- **black**: Code formatting
- **flake8**: Linting
- **mypy**: Type checking

## Development

### Running Tests

```bash
uv run pytest
```

### Code Formatting

```bash
uv run black src/
uv run flake8 src/
uv run mypy src/
```

### Adding Dependencies

```bash
# Add runtime dependency
uv add <package-name>

# Add development dependency
uv add --dev <package-name>
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Run the test suite
6. Submit a pull request

## License

This project is part of the SlimRAG ecosystem. Please refer to the main project for licensing information.

## Support

For issues and questions, please refer to the main SlimRAG project documentation and support channels.
