"""
OpenResearcher Environment - Research question answering with web search

A single-turn evaluation environment with 6,102 research questions requiring
web search. Agents must research questions, then submit answers with explanation
and confidence. Answers are graded by an LLM judge (gpt-5-mini).
"""

import asyncio

import pandas as pd
import openai
from pydantic import BaseModel, Field
from typing import Dict, List

from tavily import AsyncTavilyClient

from openreward.environments import Environment, JSONObject, TextBlock, ToolOutput, tool

from constants import OPENRESEARCHER_PARQUET


# Grader prompt template for LLM-based answer evaluation
GRADER_PROMPT_TEMPLATE = """You are evaluating whether an agent's answer to a research question is correct.

Question: {question}

Correct Answer: {correct_answer}

Agent's Response:
- Explanation: {explanation}
- Exact Answer: {exact_answer}
- Confidence: {confidence}

Task: Determine if the agent's "Exact Answer" is semantically equivalent to the correct answer.

Consider:
1. Does the exact answer capture the key factual content?
2. Are minor formatting/phrasing differences acceptable? (e.g., "Paris" vs "Paris, France")
3. Is the answer factually accurate according to the correct answer provided?
4. For numerical answers, allow small rounding differences
5. For multi-part answers, check if all key components are present

Provide a brief analysis (2-3 sentences), then conclude with either "CORRECT" or "INCORRECT" on a new line."""


class OpenResearcherTaskSpec(BaseModel):
    """Task specification for OpenResearcher environment"""
    qid: str  # Question ID
    question: str  # Research question
    answer: str  # Ground truth answer


class WebSearchInput(BaseModel):
    """Parameters for web_search tool"""
    query: str = Field(..., description="Search query for research")


class FetchUrlInput(BaseModel):
    """Parameters for fetch_url tool"""
    url: str = Field(..., description="URL to fetch and extract content from")


class SubmitAnswerParams(BaseModel):
    """Parameters for submit_answer tool"""
    explanation: str = Field(
        ...,
        description="Your detailed reasoning and sources (2-4 sentences)"
    )
    exact_answer: str = Field(
        ...,
        description="The precise answer to the research question (concise)"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Your confidence level (0.0 to 1.0)"
    )


def load_openresearcher_data() -> Dict[str, List[Dict]]:
    """
    Load OpenResearcher dataset from parquet file.

    Returns:
        Dict with "train" split containing list of task dicts

    Raises:
        FileNotFoundError: If parquet file not found at expected path
        ValueError: If data loading/parsing fails
    """
    print(f"Loading OpenResearcher data from: {OPENRESEARCHER_PARQUET}")

    if not OPENRESEARCHER_PARQUET.exists():
        raise FileNotFoundError(
            f"OpenResearcher parquet not found at {OPENRESEARCHER_PARQUET}. "
            f"Please ensure data is uploaded to /orwd_data/openresearcher/ or "
            f"available locally. See DATA_UPLOAD.md for instructions."
        )

    df = pd.read_parquet(OPENRESEARCHER_PARQUET)

    # Validate expected columns
    required_cols = {"qid", "question", "answer"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"Parquet file missing required columns. "
            f"Expected: {required_cols}, Found: {set(df.columns)}"
        )

    tasks = []
    for idx, row in df.iterrows():
        try:
            tasks.append({
                "qid": str(row['qid']),  # Ensure string
                "question": str(row['question']),
                "answer": str(row['answer']),
            })
        except Exception as e:
            print(f"Warning: Failed to process task {idx}: {e}")
            continue

    print(f"Successfully loaded {len(tasks)} tasks from seed_42 train split")
    return {"train": tasks}


# Load dataset once at module level (AIME pattern)
ALL_DATA = load_openresearcher_data()


class OpenResearcher(Environment):
    """
    OpenResearcher environment: research questions with web search + LLM grading.

    Agent workflow:
    1. Receives a research question requiring web search
    2. Uses web_search tool to find information
    3. Uses fetch_url tool to get detailed content from URLs
    4. Submits answer with explanation, exact_answer, and confidence
    5. Answer is graded by gpt-5-mini comparing to correct answer
    6. Receives reward (1.0 correct, 0.0 incorrect) and feedback
    """

    def __init__(self, task_spec: JSONObject, secrets: dict[str, str] = {}) -> None:
        """
        Initialize OpenResearcher environment instance.

        Args:
            task_spec: Task specification with qid, question, answer
            secrets: Must contain "openai_api_key" for grading and "tavily_api_key" for search

        Raises:
            ValueError: If required API keys missing or task_spec invalid
        """
        super().__init__(task_spec)
        self.config = OpenResearcherTaskSpec.model_validate(task_spec)

        # Require OpenAI API key for grader - fail fast if missing
        openai_api_key = secrets.get("openai_api_key")
        if not openai_api_key:
            raise ValueError(
                "openai_api_key required in secrets parameter for LLM grading. "
                "Pass secrets={'openai_api_key': 'sk-...', 'tavily_api_key': 'tvly-...'} when creating session."
            )

        # Require Tavily API key for web search - fail fast if missing
        tavily_api_key = secrets.get("tavily_api_key")
        if not tavily_api_key:
            raise ValueError(
                "tavily_api_key required in secrets parameter for web search. "
                "Pass secrets={'openai_api_key': 'sk-...', 'tavily_api_key': 'tvly-...'} when creating session."
            )

        self.openai_client = openai.AsyncClient(api_key=openai_api_key)
        self.tavily_client = AsyncTavilyClient(api_key=tavily_api_key)

    @classmethod
    def list_splits(cls) -> list[str]:
        """Return available data splits"""
        return ["train"]

    @classmethod
    def list_tasks(cls, split: str) -> list[JSONObject]:
        """
        List all tasks for a given split.

        Args:
            split: Data split name (only "train" available)

        Returns:
            List of task specifications (qid, question, answer)

        Raises:
            ValueError: If split is unknown
        """
        if split != "train":
            raise ValueError(f"Unknown split: {split}. Available splits: train")

        # Return all task fields including answer (needed for grading)
        return [
            {
                "qid": task["qid"],
                "question": task["question"],
                "answer": task["answer"],
            }
            for task in ALL_DATA["train"]
        ]

    async def get_prompt(self) -> list[TextBlock]:
        """
        Generate prompt for the agent.

        Returns:
            List containing single TextBlock with question and instructions
        """
        prompt_text = f"""Research Question: {self.config.question}

Your task is to research this question using web search and provide a comprehensive answer."""

        return [TextBlock(type="text", text=prompt_text)]

    async def _tavily_with_retry(self, label: str, call, *, max_attempts: int = 4):
        """Call Tavily with exponential backoff, re-raising on persistent failure.

        A genuinely-down dependency (exhausted quota, auth error) exhausts the
        retries and re-raises, so the SDK marks the call ToolFailed and ends the
        rollout. `call` returns a fresh awaitable on each attempt.
        """
        last_exc: Exception | None = None
        for attempt in range(max_attempts):
            try:
                return await call()
            except Exception as e:
                last_exc = e
                if attempt < max_attempts - 1:
                    wait = min(2 ** attempt, 30)
                    print(f"TAVILY ERROR: {label} | {e} | retry in {wait}s (attempt {attempt + 1}/{max_attempts})")
                    await asyncio.sleep(wait)
        assert last_exc is not None
        raise last_exc

    @tool
    async def web_search(self, params: WebSearchInput) -> ToolOutput:
        """
        Search the web using Tavily. Returns search results with titles, URLs, and snippets.
        Use fetch_url tool to get full content from specific URLs if needed.
        """
        response = await self._tavily_with_retry(
            f"search({params.query!r})",
            lambda: self.tavily_client.search(
                query=params.query,
                search_depth="basic",
                max_results=5
            ),
        )

        # reward stays None throughout: retrieval is not a scoring event, and 0.0
        # would assert a zero score on every search. submit_answer does the scoring.
        results = response.get("results", [])
        if not results:
            return ToolOutput(
                blocks=[TextBlock(type="text", text="No search results found.")],
                metadata={"query": params.query, "results": []},
                reward=None,
                finished=False
            )

        # Build display text
        display_parts = [f"Search results for: {params.query}\n"]
        for i, result in enumerate(results, 1):
            title = result.get("title", "No title")
            url = result.get("url", "")
            snippet = result.get("content", "")
            display_parts.append(f"{i}. {title}\n   URL: {url}\n   {snippet}\n")

        display_text = "\n".join(display_parts)

        return ToolOutput(
            blocks=[TextBlock(type="text", text=display_text)],
            metadata={
                "query": params.query,
                "results": results,
                "count": len(results)
            },
            reward=None,
            finished=False
        )

    @tool
    async def fetch_url(self, params: FetchUrlInput) -> ToolOutput:
        """
        Fetch and return the full text content from a specific URL using Tavily's extract method.
        Use this after web_search to get complete information from a page.
        """
        response = await self._tavily_with_retry(
            f"extract({params.url!r})",
            lambda: self.tavily_client.extract(urls=[params.url]),
        )

        # reward stays None throughout: see web_search.
        results = response.get("results", [])
        if not results:
            # No result object at all — usually a fetch failure (DNS/timeout/
            # blocked) or an unsupported URL, which the agent can recover from
            # by picking a different source.
            return ToolOutput(
                blocks=[TextBlock(type="text", text=(
                    f"No content extracted from {params.url}. The URL may be "
                    f"unreachable, blocked, or invalid. Try a different source."
                ))],
                metadata={"url": params.url, "results": []},
                reward=None,
                finished=False
            )

        # Get the first result (we only passed one URL)
        result = results[0]
        raw_content = result.get("raw_content", "")

        # Truncate if too long
        max_length = 8000
        if len(raw_content) > max_length:
            raw_content = raw_content[:max_length] + "...\n[Content truncated]"

        return ToolOutput(
            blocks=[TextBlock(type="text", text=f"Content from {params.url}:\n\n{raw_content}")],
            metadata={
                "url": params.url,
                "length": len(raw_content)
            },
            reward=None,
            finished=False
        )

    @staticmethod
    def _parse_verdict(grading_text: str) -> bool:
        """Read the grader's CORRECT/INCORRECT verdict from its final line.

        The grader is prompted to end with the verdict alone on a new line, so we
        scan upward from the end for the first line that resolves to one. The
        INCORRECT check runs first because CORRECT is its suffix.

        Searching the whole response for "CORRECT" cannot work: "INCORRECT"
        contains it, so an analysis calling an answer "not incorrect" — ordinary
        phrasing for a near-miss the grader still accepts — scores a CORRECT
        verdict 0.0.
        """
        for line in reversed(grading_text.splitlines()):
            token = line.strip().strip("*_#`.:!-").strip().upper()
            if not token:
                continue
            if token.endswith("INCORRECT"):
                return False
            if token.endswith("CORRECT"):
                return True

        # No verdict line at all — treat as incorrect, but say so, since a silent
        # 0.0 here is indistinguishable from a genuinely wrong answer.
        print(f"GRADER WARNING: no CORRECT/INCORRECT verdict found in grader response: {grading_text[:200]!r}")
        return False

    async def _grade_answer(
        self,
        explanation: str,
        exact_answer: str,
        confidence: float
    ) -> Dict:
        """
        Use LLM grader to evaluate answer correctness.

        Args:
            explanation: Agent's reasoning
            exact_answer: Agent's submitted answer
            confidence: Agent's confidence score

        Returns:
            Dict with keys: is_correct, grading_response, confidence

        Note: Uses gpt-5-mini with no temperature parameter (as per CLAUDE.md)
        """
        grader_prompt = GRADER_PROMPT_TEMPLATE.format(
            question=self.config.question,
            correct_answer=self.config.answer,
            explanation=explanation,
            exact_answer=exact_answer,
            confidence=confidence
        )

        # Use gpt-5-mini as recommended for graders (cost-effective, reliable)
        # IMPORTANT: No temperature parameter (per CLAUDE.md guidelines)
        response = await self.openai_client.chat.completions.create(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": grader_prompt}],
        )

        grading_text = response.choices[0].message.content or ""
        is_correct = self._parse_verdict(grading_text)

        return {
            "is_correct": is_correct,
            "grading_response": grading_text,
            "confidence": confidence
        }

    @tool
    async def submit_answer(self, params: SubmitAnswerParams) -> ToolOutput:
        """
        Submit your final answer to the research question.

        This tool grades your answer using an LLM judge and returns a reward.
        The episode ends after calling this tool.

        Args:
            explanation: Your reasoning and sources (2-4 sentences)
            exact_answer: The precise answer to the question
            confidence: Your confidence level (0.0 to 1.0)

        Returns:
            ToolOutput with grading result, reward, and feedback
        """
        # Grade the answer using LLM judge
        grading_result = await self._grade_answer(
            params.explanation,
            params.exact_answer,
            params.confidence
        )

        reward = 1.0 if grading_result["is_correct"] else 0.0
        result_status = "✅ Correct" if grading_result["is_correct"] else "❌ Incorrect"

        # Format display output for the agent
        display_text = f"""{result_status}

Grading Analysis:
{grading_result['grading_response']}

Your Confidence: {params.confidence:.2f}
Reward: {reward:.1f}

Expected Answer: {self.config.answer}
Your Answer: {params.exact_answer}"""

        return ToolOutput(
            blocks=[TextBlock(type="text", text=display_text)],
            metadata={
                "qid": self.config.qid,
                "is_correct": grading_result["is_correct"],
                "grading_response": grading_result["grading_response"],
                "submitted_answer": params.exact_answer,
                "submitted_explanation": params.explanation,
                "confidence": params.confidence,
                "correct_answer": self.config.answer,  # For analysis
                "question": self.config.question,
            },
            reward=reward,
            finished=True
        )
