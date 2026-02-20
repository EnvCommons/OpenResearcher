# OpenResearcher Environment

A single-turn evaluation environment for the OpenResearcher dataset, featuring 6,102 research questions that require web search and multi-hop reasoning.

## Overview

OpenResearcher is an OpenReward environment that evaluates AI agents' ability to research complex questions using web search tools. Agents are presented with challenging research questions spanning multiple domains and must:

1. Use `web_search` to find relevant information across multiple queries
2. Use `fetch_url` to extract detailed content from promising URLs
3. Synthesize information and submit answers with `submit_answer`

Answers are evaluated using an LLM judge (gpt-5-mini) that performs semantic matching against ground truth answers.

## Key Features

- **6,102 Research Questions** from OpenResearcher-Dataset (seed_42)
- **Web Search Integration** via Tavily API (proven, production-ready)
- **Semantic Answer Grading** using gpt-5-mini (handles variations in phrasing/formatting)
- **Single-Turn Evaluation** for simplicity and reliability
- **Binary Rewards** (1.0 for correct, 0.0 for incorrect)

## Dataset

- **Source:** [OpenResearcher/OpenResearcher-Dataset](https://huggingface.co/datasets/OpenResearcher/OpenResearcher-Dataset) (HuggingFace)
- **Configuration:** seed_42
- **Split:** train
- **Size:** 6,102 research questions
- **Domains:** Diverse (structured data, technical research, historical facts, art history, legislative research, etc.)

### Example Questions

1. "What is the grand finalist where the winner is Collingwood among preseason and night series Australian Football League 1977 1987?"
2. "Identify the method for personalizing image synthesis models to user-provided visual concepts..."
3. "Four pharaohs of a golden age of ancient Egypt's pyramid-building dynasty..."

## Installation

### Prerequisites

- Python 3.11+
- OpenAI API key (for answer grading)
- Tavily API key (for web search)

### Local Development

1. **Clone the repository:**
   ```bash
   git clone https://github.com/EnvCommons/openresearcher.git
   cd openresearcher
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare dataset:**
   ```bash
   python prepare_dataset.py
   ```

   This downloads the dataset from HuggingFace and creates `openresearcher_seed42.parquet` locally.

4. **Set API keys:**
   ```bash
   export OPENAI_API_KEY='sk-...'
   export TAVILY_API_KEY='tvly-...'
   ```

5. **Run server:**
   ```bash
   python server.py
   ```

   Server will start on `http://0.0.0.0:8080`

6. **Test agent** (in another terminal):
   ```bash
   export OPENAI_API_KEY='sk-...'
   export TAVILY_API_KEY='tvly-...'
   python test_agent.py
   ```

### Docker Deployment

1. **Build image:**
   ```bash
   docker build -t openresearcher:latest .
   ```

2. **Run container:**
   ```bash
   docker run -p 8080:8080 openresearcher:latest
   ```

**Note:** For production deployment, upload the dataset to OpenReward cloud storage. See [DATA_UPLOAD.md](DATA_UPLOAD.md) for detailed instructions.

## Environment API

### Tools

#### `web_search(query: str)`

Search the web using Tavily API.

**Parameters:**
- `query` (str): Search query string

**Returns:**
- Search results with titles, URLs, and content snippets
- Max 5 results per search
- Formatted as numbered list

**Example:**
```python
web_search(query="ancient Egypt pyramid dynasty pharaohs")
```

#### `fetch_url(url: str)`

Fetch full content from a URL using Tavily's extract method.

**Parameters:**
- `url` (str): URL to fetch

**Returns:**
- Extracted text content
- Truncated to 8000 characters max
- Raw content from the page

**Example:**
```python
fetch_url(url="https://example.com/article")
```

#### `submit_answer(explanation: str, exact_answer: str, confidence: float)`

Submit final answer with explanation.

**Parameters:**
- `explanation` (str): Detailed reasoning and sources cited (2-4 sentences)
- `exact_answer` (str): Precise, concise answer to the question
- `confidence` (float): Confidence level between 0.0 and 1.0

**Returns:**
- Grading result (correct/incorrect)
- Reward: 1.0 (correct) or 0.0 (incorrect)
- Feedback with grading analysis
- Episode ends (`finished=True`)

**Example:**
```python
submit_answer(
    explanation="Based on multiple sources about ancient Egypt, the four pharaohs of the pyramid-building golden age were Khufu, Khafre, Menkaure, and Sneferu, who built the Great Pyramids of Giza during the Fourth Dynasty.",
    exact_answer="Khufu, Khafre, Menkaure, Sneferu",
    confidence=0.9
)
```

## Grading

Answers are evaluated using **gpt-5-mini** as an LLM judge. The grader:

1. Compares the submitted `exact_answer` to the ground truth answer
2. Considers **semantic equivalence** (not just exact string matching)
3. Allows minor formatting/phrasing differences (e.g., "Paris" vs "Paris, France")
4. Checks factual accuracy
5. Handles numerical answers with reasonable precision
6. Validates all key components are present in multi-part answers

**Reward Structure:**
- **1.0** - Answer is semantically correct
- **0.0** - Answer is incorrect or incomplete

## Usage Example

```python
import asyncio
from openreward import AsyncOpenReward

async def main():
    or_client = AsyncOpenReward()

    # Connect to environment
    environment = or_client.environments.get(name="EnvCommons/openresearcher")

    # Get tasks
    tasks = await environment.list_tasks(split="train")
    task = tasks[0]

    # Create session with required API keys
    async with environment.session(
        task=task,
        secrets={
            "openai_api_key": "sk-...",
            "tavily_api_key": "tvly-..."
        }
    ) as session:
        # Get prompt
        prompt = await session.get_prompt()
        print(prompt[0].text)

        # Agent uses tools here...
        # 1. web_search multiple times
        # 2. fetch_url for detailed content
        # 3. submit_answer with findings

        # Example tool call
        result = await session.call_tool(
            "web_search",
            {"query": "ancient Egypt pyramid pharaohs"}
        )
        print(result.blocks[0].text)

if __name__ == "__main__":
    asyncio.run(main())
```

## File Structure

```
openresearcher/
├── openresearcher.py       # Main environment class
├── server.py               # Minimal server wrapper
├── constants.py            # Path handling logic
├── test_agent.py           # Agent testing script
├── prepare_dataset.py      # Dataset download script
├── requirements.txt        # Python dependencies
├── Dockerfile              # Container configuration
├── DATA_UPLOAD.md          # Data upload instructions
├── .gitignore              # Git ignore patterns
└── README.md               # This file
```

## Architecture

**Pattern:** Single-turn evaluation (extends `Environment`, not `CLIEnvironment`)

**Components:**
- **Data Loading:** Module-level loading from parquet file (AIME pattern)
- **Path Handling:** Supports both `/orwd_data` (production) and local paths (development)
- **Search Backend:** Tavily API for web search and URL extraction
- **Grading:** gpt-5-mini semantic matching (no temperature parameter)

**API Keys Required:**
1. `openai_api_key` - For LLM-based answer grading
2. `tavily_api_key` - For web search and URL fetching

Both keys are **required** and validated at environment initialization (fail-fast pattern).

## Testing

### Syntax Check
```bash
python -m py_compile *.py
```

### Local Server Test
```bash
python server.py
# Should start without errors on http://0.0.0.0:8080
```

### Agent Integration Test
```bash
python test_agent.py
# Runs first task with agent interaction
# Shows tool calls, results, and final reward
```

### Docker Test
```bash
docker build -t openresearcher:test .
docker run -p 8080:8080 openresearcher:test
```

## Troubleshooting

### Error: "OpenResearcher parquet not found"

**Solution:**
1. Run `python prepare_dataset.py` to download dataset locally
2. For production, upload to `/orwd_data/openresearcher/` (see [DATA_UPLOAD.md](DATA_UPLOAD.md))

### Error: "openai_api_key required in secrets"

**Solution:**
Pass API keys via `secrets` parameter when creating session:
```python
async with environment.session(
    task=task,
    secrets={
        "openai_api_key": "sk-...",
        "tavily_api_key": "tvly-..."
    }
) as session:
    ...
```

### Error: "tavily_api_key required in secrets"

**Solution:**
Sign up for Tavily API at https://tavily.com (free tier: 1000 requests/month)

### Warning: "Failed to process task X"

**Solution:**
Some dataset rows may be malformed. The environment skips problematic rows and continues with valid tasks.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests (`python -m py_compile *.py`)
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Citation

If you use this environment in your research, please cite:

```bibtex
@misc{openresearcher-env-2024,
  title={OpenResearcher Environment for OpenReward},
  author={EnvCommons},
  year={2024},
  publisher={GitHub},
  url={https://github.com/EnvCommons/openresearcher}
}
```

For the underlying dataset:

```bibtex
@misc{openresearcher-dataset,
  title={OpenResearcher Dataset},
  author={OpenResearcher},
  year={2024},
  publisher={HuggingFace},
  url={https://huggingface.co/datasets/OpenResearcher/OpenResearcher-Dataset}
}
```

## Support

For questions or issues:
- Open an issue on [GitHub](https://github.com/EnvCommons/openresearcher/issues)
- Check [DATA_UPLOAD.md](DATA_UPLOAD.md) for data-related questions
- Review [test_agent.py](test_agent.py) for usage examples

## Related Resources

- [OpenReward Documentation](https://docs.openreward.org/)
- [OpenResearcher Dataset on HuggingFace](https://huggingface.co/datasets/OpenResearcher/OpenResearcher-Dataset)
- [Tavily API Documentation](https://docs.tavily.com/)
- [BrowseComp Environment](https://github.com/EnvCommons/BrowseComp) (similar pattern)
