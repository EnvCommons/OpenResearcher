# OpenResearcher

[![⭐ OpenReward Environment](https://img.shields.io/badge/%E2%AD%90%20OpenReward-Environment-f7e6cc)](https://openreward.ai/GeneralReasoning/OpenResearcher) [![Hugging Face Dataset](https://img.shields.io/badge/Hugging%20Face-Dataset-orange)](https://huggingface.co/datasets/OpenResearcher/OpenResearcher-Dataset)

## Description

OpenResearcher is an ORS environment for evaluating research question answering through web search. Based on the OpenResearcher dataset, agents are given diverse research questions and must use web search and URL fetching to find and synthesize answers. Search and fetch run over OpenReward's backdated web corpus (backsearch), with the cutoff fixed to the UTC date on which the session starts, so the agent sees the web as it stands that day. An LLM grader evaluates semantic correctness.

## Capabilities

- Research question answering via web search
- Multi-hop information retrieval
- Synthesizing answers from web sources

## Compute Requirements

This is a multi-turn environment with no sandbox. Agents interact through web search and URL fetching tools only, which are served by OpenReward's backsearch service over HTTP.

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

The agent uses `web_search` and `web_fetch` to research, then calls `submit_answer` with an explanation, exact answer, and confidence score. The LLM grader evaluates semantic equivalence, accepting minor formatting and phrasing differences.

## Data

Data consists of a single Parquet file (`openresearcher_seed42.parquet`) containing 6,102 research questions with ground truth answers. Each instance includes a question ID, the research question text, and the correct answer.

Source: [OpenResearcher/OpenResearcher-Dataset](https://huggingface.co/datasets/OpenResearcher/OpenResearcher-Dataset) (seed_42 configuration)

## Tools

| Tool | Description |
|------|-------------|
| `web_search` | Search OpenReward's backdated web corpus as of the session's start date. Returns up to 8 hits, each with a title, URL and text snippet, fanned out over the backend's default corpora (news, SEC filings, Wikipedia, general web, live captures and arXiv). Supports `allowed_domains` or `blocked_domains` (not both). |
| `web_fetch` | Fetch the archived text of a URL as it existed on or before the session's start date, applying a caller-supplied prompt. Returns up to 100,000 characters; cross-host redirects come back as a `REDIRECT DETECTED` notice to re-fetch. |
| `submit_answer` | Submit explanation, exact answer, and confidence score for LLM grading. Ends the episode. |

The search tools are pinned to the backdated corpus (`BackSearchToolset`) rather than a switchable live-web provider, so the point-in-time guarantee cannot be turned off by an environment variable. The cutoff is set once per session in the environment's constructor and read by the toolset on every call.

## Time Horizon

OpenResearcher is a multi-turn environment. Agents search the web, fetch URLs for detailed content, and submit a final answer when ready.

## Environment Difficulty

[Put environment difficulty here once available]

## Other Environment Requirements

- **OpenAI API key**: Required for LLM-based answer grading via gpt-5-mini

Pass via `secrets={"openai_api_key": "..."}`.

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
