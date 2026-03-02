# OpenResearcher

[![⭐ OpenReward](https://img.shields.io/badge/%E2%AD%90%20OpenReward-Environment-f7e6cc)](https://openreward.ai/GeneralReasoning/OpenResearcher)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Dataset-orange)](https://huggingface.co/datasets/OpenResearcher/OpenResearcher-Dataset)

## Description

OpenResearcher is an environment for evaluating research question answering through web search. Based on the OpenResearcher dataset, agents are given diverse research questions and must use web search and URL fetching to find and synthesize answers. An LLM grader evaluates semantic correctness.

## Capabilities

- Research question answering via web search
- Multi-hop information retrieval
- Synthesizing answers from web sources

## Compute Requirements

Agents are given a standard environment with no sandbox or file system access.

## License

[MIT](https://opensource.org/licenses/MIT)

## Tasks

One split: `train` (6,102 research questions)

## Reward Structure

Multi-turn (agent searches, then submits). Agent uses `web_search` and `fetch_url` to research, then calls `submit_answer` with explanation, exact answer, and confidence. LLM grader (gpt-5-mini) evaluates semantic correctness. Binary reward: 1.0 if correct, 0.0 if incorrect.

## Data

`openresearcher_seed42.parquet` sourced from [HuggingFace OpenResearcher/OpenResearcher-Dataset](https://huggingface.co/datasets/OpenResearcher/OpenResearcher-Dataset) (seed_42 config). Stored on the OpenReward platform.

## Tools

- **`web_search`**: Search the web via Tavily API, returns top 5 results
- **`fetch_url`**: Fetch and extract text content from a URL, max 8000 chars
- **`submit_answer`**: Submit explanation, exact answer, and confidence for grading

## Time Horizon

Multi-turn. Agents search the web, fetch URLs, and submit a final answer.

## Environment Difficulty

Tasks span diverse domains including technical research, historical facts, art history, and legislative research, requiring multi-hop reasoning and web search proficiency.

## Other Environment Requirements

OpenAI API key required for grading. Tavily API key required for web search. Pass via `secrets={"openai_api_key": "...", "tavily_api_key": "..."}`.

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
