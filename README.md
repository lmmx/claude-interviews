# interviewer

Explore Claude interview responses with semantic search.

Uses the [Anthropic/AnthropicInterviewer](https://huggingface.co/datasets/Anthropic/AnthropicInterviewer) dataset.

## Setup
```bash
pdm install
```

## Usage

1. Generate embeddings (run once):
```bash
pdm run interviewer-embed
```

2. Start the server:
```bash
pdm run interviewer-serve
```

3. Open http://localhost:8000
```

---

**.gitignore**
```
.*_cache
output/