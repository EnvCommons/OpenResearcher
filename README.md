# OpenResearcher

[![⭐ OpenReward Environment](https://img.shields.io/badge/%E2%AD%90%20OpenReward-Environment-f7e6cc)](https://openreward.ai/GeneralReasoning/OpenResearcher) [![Hugging Face Dataset](https://img.shields.io/badge/Hugging%20Face-Dataset-orange)](https://huggingface.co/datasets/OpenResearcher/OpenResearcher-Dataset)

## Description

OpenResearcher is an ORS environment for evaluating research question answering through web search. Based on the OpenResearcher dataset, agents are given diverse research questions and must use web search and URL fetching to find and synthesize answers. An LLM grader evaluates semantic correctness.

## Capabilities

- Research question answering via web search
- Multi-hop information retrieval
- Synthesizing answers from web sources

## Compute Requirements

This is a multi-turn environment with no sandbox. Agents interact through web search and URL fetching tools only.

## License

[MIT](https://opensource.org/licenses/MIT)

## Tasks

There is one split in this environment:

- **Train**: 6,102 research questions

Each task presents a research question requiring web search to answer. Questions span technical research, historical facts, art history, legislative research, and other domains.

## Reward Structure

This is a multi-turn environment with binary reward:

- **1.0** — Correct answer (semantically equivalent to the reference, as judged by gpt-5-mini)
- **0.0** — Incorrect answer

The agent uses `web_search` and `fetch_url` to research, then calls `submit_answer` with an explanation, exact answer, and confidence score. The LLM grader evaluates semantic equivalence, accepting minor formatting and phrasing differences.

## Data

Data consists of a single Parquet file (`openresearcher_seed42.parquet`) containing 6,102 research questions with ground truth answers. Each instance includes a question ID, the research question text, and the correct answer.

Source: [OpenResearcher/OpenResearcher-Dataset](https://huggingface.co/datasets/OpenResearcher/OpenResearcher-Dataset) (seed_42 configuration)

## Tools

| Tool | Description |
|------|-------------|
| `web_search` | Search the web via Tavily API. Returns top 5 results with titles, URLs, and snippets. |
| `fetch_url` | Fetch and extract text content from a URL. Truncates to 8,000 characters. |
| `submit_answer` | Submit explanation, exact answer, and confidence score for LLM grading. Ends the episode. |

Note that the `fetch_url` and `web_search` tools require Tavily, but are optional. If you want to use a different provider for search you can exclude these tools and use external tools instead.

## Time Horizon

OpenResearcher is a multi-turn environment. Agents search the web, fetch URLs for detailed content, and submit a final answer when ready.

## Environment Difficulty

[Put environment difficulty here once available]

## Other Environment Requirements

- **OpenAI API key**: Required for LLM-based answer grading via gpt-5-mini
- **Tavily API key**: Required for web search and URL content extraction

Pass via `secrets={"openai_api_key": "...", "tavily_api_key": "..."}`.

## Safety

Agents in OpenResearcher answer research questions using web search in a standard environment. The environment does not present direct safety risks.

## Citations

```bibtex
@article{zheng2024openresearcher,
  title={OpenResearcher: Unleashing AI for Accelerated Scientific Research},
  author={Zheng, Yuxiang and Sun, Shichao and Qiu, Lin and Ru, Dongyu and Jiayang, Cheng and Li, Xuefeng and Lin, Jifan and Wang, Binjie and Luo, Yun and Pan, Renjie and others},
  journal={arXiv preprint arXiv:2408.06941},
  year={2024}
}
```
