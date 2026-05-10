# 语义级论文查重系统 (Semantic Thesis Plagiarism Detection System)

A comprehensive, semantic-level plagiarism detection system for academic theses, combining deep semantic analysis, knowledge graph-based structural comparison, and AI-generated content detection.

## Features

- **Semantic Similarity Analysis** — BERT-based deep semantic encoding and similarity computation between thesis segments, capturing meaning beyond surface-level text matching.
- **Knowledge Graph Comparison** — Builds structural knowledge graphs from thesis content and compares them using graph-theoretic algorithms (NetworkX) to detect structural plagiarism.
- **AI-Generated Content Detection** — Multi-faceted detection of AI-written content using statistical feature analysis, coherence modeling, and linguistic fingerprinting.
- **Discipline-Specific Configuration** — Pluggable configuration models for different academic disciplines, allowing custom thresholds, weighting schemes, and domain-specific dictionaries.
- **RESTful API** — FastAPI-based web service providing endpoints for submission upload, plagiarism analysis, report retrieval, and batch processing.
- **Redis-Backed Caching** — High-performance caching layer for embeddings and intermediate results, enabling incremental re-analysis and rapid repeat checks.

## Architecture

```
thesis-plagiarism-check/
├── src/thesischeck/
│   ├── core/
│   │   ├── semantic/         # BERT encoding, similarity scoring, knowledge graph
│   │   ├── ai_detection/     # AI-generated text detection pipeline
│   │   ├── config/           # Discipline-specific configuration models
│   │   │   └── disciplines/  # Per-discipline config files
│   │   └── report/           # Report generation and formatting
│   ├── pipeline/             # Text preprocessing and analysis orchestration
│   ├── api/                  # FastAPI REST API endpoints
│   └── cache/                # Redis caching layer
├── tests/                    # Test suite
│   └── fixtures/             # Test data and mock objects
└── pyproject.toml            # Project configuration and build metadata
```

## Quick Start

### Prerequisites

- Python 3.11 or later
- Redis server (for caching, optional but recommended)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd thesis-plagiarism-check

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e ".[dev]"
```

### Running the API Server

```bash
uvicorn src.thesischeck.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Running Tests

```bash
pytest
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST   | `/api/v1/check`          | Submit a thesis for plagiarism analysis |
| GET    | `/api/v1/status/{task_id}` | Check analysis task status |
| GET    | `/api/v1/report/{task_id}` | Retrieve completed analysis report |
| POST   | `/api/v1/compare`        | Compare two thesis documents directly |
| GET    | `/api/v1/health`          | Health check endpoint |

## License

MIT