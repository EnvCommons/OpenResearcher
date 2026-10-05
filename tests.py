import asyncio

import pytest
from openreward.toolsets._web_common import WebFetchParams, WebSearchParams
from openreward.tools.web import WebToolResult, format_search_output

import openresearcher
from openresearcher import OpenResearcher, OpenResearcherBackSearch, SubmitAnswerParams, _is_blocked_url

SECRETS = {"openai_api_key": "test-key", "api_key": "test-key"}
TASK = {"qid": "q1", "question": "What is the capital of France?", "answer": "Paris"}
ANSWER = SubmitAnswerParams(explanation="It is.", exact_answer="Paris", confidence=0.9)


@pytest.mark.parametrize("url", [
    "https://huggingface.co/datasets/OpenResearcher/OpenResearcher-Dataset",
    "https://huggingface.co/datasets/callanwu/WebWalkerQA/viewer/default/silver",
    "http://www.huggingface.co/datasets/miromind-ai/MiroVerse-v0.1",
    "huggingface.co/datasets/x/y",
    "https://HuggingFace.co/Datasets/x/y",
    "https://huggingface.co/api/datasets/x/y",
    "https://hf.co/datasets/x/y",
    "https://hf-mirror.com/datasets/x/y",
    "https://datasets-server.huggingface.co/rows?dataset=x&config=default&split=train",
    "https://www.modelscope.cn/datasets/x/y",
    "https://huggingface.co/datasets",
])
def test_dataset_urls_blocked(url):
    assert _is_blocked_url(url)


@pytest.mark.parametrize("url", [
    "https://huggingface.co/papers/2408.06941",
    "https://huggingface.co/meta-llama/Llama-3.1-8B",
    "https://huggingface.co/datasetsfoo",
    "https://en.wikipedia.org/wiki/Datasets",
    "https://example.com/datasets/x",
    "https://arxiv.org/abs/2408.06941",
])
def test_other_urls_allowed(url):
    assert not _is_blocked_url(url)


def _toolset():
    return OpenResearcherBackSearch(OpenResearcher(TASK, SECRETS))


def test_search_drops_dataset_hits(monkeypatch):
    hits = [
        {"title": "viewer", "url": "https://huggingface.co/datasets/a/b/viewer", "snippet": "Q: ... A: secret"},
        {"title": "wiki", "url": "https://en.wikipedia.org/wiki/Paris", "snippet": "Paris is"},
        {"title": "rows", "url": "https://datasets-server.huggingface.co/rows?dataset=a", "snippet": "secret"},
    ]

    async def fake_search(*, query, include_snippets, **kwargs):
        return WebToolResult.success(
            format_search_output(query, hits, include_snippets=include_snippets),
            {"query": query, "hits": hits, "mode": "x"},
        )

    monkeypatch.setattr(openresearcher, "run_search", fake_search)
    out = asyncio.run(_toolset().web_search(WebSearchParams(query="capital of France")))
    text = out.blocks[0].text
    assert "wikipedia.org" in text
    assert "huggingface.co" not in text and "secret" not in text
    assert [h["url"] for h in out.metadata["hits"]] == ["https://en.wikipedia.org/wiki/Paris"]
    assert not out.finished


def test_fetch_refuses_dataset_pages(monkeypatch):
    async def fake_fetch(**kwargs):
        raise AssertionError("blocked URL must not be fetched")

    monkeypatch.setattr(openresearcher, "run_fetch", fake_fetch)
    out = asyncio.run(_toolset().web_fetch(WebFetchParams(
        url="https://huggingface.co/datasets/callanwu/WebWalkerQA/viewer", prompt="answer?")))
    assert out.metadata["error"] == "blocked-url"
    assert out.reward == 0.0 and not out.finished


def test_fetch_allows_other_pages(monkeypatch):
    async def fake_fetch(*, url, **kwargs):
        return WebToolResult.success(f"page {url}", {"url": url})

    monkeypatch.setattr(openresearcher, "run_fetch", fake_fetch)
    out = asyncio.run(_toolset().web_fetch(WebFetchParams(url="https://huggingface.co/papers/1", prompt="p")))
    assert out.blocks[0].text == "page https://huggingface.co/papers/1"


def _env_with_grader(monkeypatch, calls):
    env = OpenResearcher(TASK, SECRETS)

    async def fake_grade(explanation, exact_answer, confidence):
        calls.append(exact_answer)
        await asyncio.sleep(0.01)
        return {"is_correct": True, "grading_response": "CORRECT", "confidence": confidence}

    monkeypatch.setattr(env, "_grade_answer", fake_grade)
    return env


def test_repeat_submit_is_not_graded(monkeypatch):
    calls = []
    env = _env_with_grader(monkeypatch, calls)

    async def run():
        first = await env.submit_answer(ANSWER)
        second = await env.submit_answer(ANSWER)
        return first, second

    first, second = asyncio.run(run())
    assert first.reward == 1.0 and first.finished
    assert second.reward == 0.0 and not second.finished
    assert calls == ["Paris"]


def test_concurrent_submits_grade_once(monkeypatch):
    calls = []
    env = _env_with_grader(monkeypatch, calls)

    async def run():
        return await asyncio.gather(env.submit_answer(ANSWER), env.submit_answer(ANSWER))

    results = asyncio.run(run())
    assert calls == ["Paris"]
    assert sorted(r.finished for r in results) == [False, True]
    assert sum(r.reward for r in results) == 1.0


def test_grader_failure_leaves_submit_open(monkeypatch):
    env = OpenResearcher(TASK, SECRETS)
    attempts = []

    async def flaky_grade(explanation, exact_answer, confidence):
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("Grader response had no CORRECT/INCORRECT verdict")
        return {"is_correct": False, "grading_response": "INCORRECT", "confidence": confidence}

    monkeypatch.setattr(env, "_grade_answer", flaky_grade)
    with pytest.raises(RuntimeError):
        asyncio.run(env.submit_answer(ANSWER))
    out = asyncio.run(env.submit_answer(ANSWER))
    assert out.finished and out.reward == 0.0


def test_tasks_without_options_excluded():
    tasks = OpenResearcher.list_tasks("train")
    qids = [t["qid"] for t in tasks]
    assert not set(qids) & openresearcher.EXCLUDED_QIDS
    assert len(qids) == len(set(qids))
    assert qids == [t["qid"] for t in OpenResearcher.list_tasks("train")]
