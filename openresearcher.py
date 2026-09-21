"""
OpenResearcher Environment - Research question answering with web search

A single-turn evaluation environment with 6,102 research questions requiring
web search. Agents must research questions, then submit answers with explanation
and confidence. Answers are graded by an LLM judge (gpt-5-mini).

Search and fetch go through OpenReward's backdated web corpus (backsearch).
The cutoff is the UTC date on which the session was created, so the agent sees
the web as it stands today, and every search fans out over the backend's
default corpora (news, SEC filings, Wikipedia, general web, live captures)
rather than a single source.
"""

import asyncio
from datetime import datetime, timezone

import pandas as pd
import openai
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional


from openreward.environments import Environment, JSONObject, TextBlock, ToolOutput, tool
from openreward.toolsets import BackSearchToolset
from openreward.toolsets._web_common import WebFetchParams, WebSearchParams, to_tool_output
from openreward.tools.web import FETCH_DESCRIPTION, SEARCH_DESCRIPTION, run_fetch, run_search
from openreward.web_service import WebServiceConfig

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


def today_utc_iso() -> str:
    """Today's date in UTC as ISO ``YYYY-MM-DD`` — the backsearch cutoff.

    UTC rather than the server's local date so every replica of the env agrees
    on the cutoff regardless of the timezone it happens to run in.
    """
    return datetime.now(timezone.utc).date().isoformat()


class OpenResearcherBackSearch(BackSearchToolset):
    """BackSearchToolset with three adjustments, all backdating-preserving.

    First, the session's secrets (``api_key`` / ``openreward_api_key``) are
    consulted when building the web-service config, falling back to the process
    environment's ``OPENREWARD_API_KEY`` — the stock toolset reads the process
    env only. Second, ``web_search`` passes ``include_snippets=True`` so results
    carry text snippets instead of bare titles and URLs, letting the agent
    triage hits without a fetch per candidate. Third, fatal backend errors (a
    missing key, an exhausted quota) raise ``SearchBackendUnavailable`` instead
    of becoming tool output: handed back as text, the agent would re-issue a
    dead call until the turn cap and the rollout would score 0.0 as though the
    model had answered wrongly, rather than being discarded as an
    infrastructure failure.

    The cutoff still resolves through the parent's ``_current_as_of``
    (``env.web_as_of``, the UTC date the session was created) on every call.
    No ``corpus`` is pinned, so the backend fans out over its default corpora
    (news, SEC filings, Wikipedia, general web, live captures) — naming corpora
    *replaces* that set rather than extending it, and a single-corpus pin is
    exactly what this environment must avoid. Unlike ``WebToolset``, this
    toolset cannot be switched to a live-web provider by an environment
    variable.
    """

    def __init__(self, env: Optional[Any] = None, **kwargs: Any) -> None:
        if kwargs.get("config") is None:
            secrets = getattr(env, "search_secrets", None)
            kwargs["config"] = WebServiceConfig.from_env(secrets)
        super().__init__(env, **kwargs)

    @tool
    async def web_search(self, params: WebSearchParams) -> ToolOutput:
        result = await run_search(
            query=params.query,
            as_of=self._current_as_of(),
            allowed_domains=params.allowed_domains,
            blocked_domains=params.blocked_domains,
            config=self.config,
            include_snippets=True,
        )
        return to_tool_output(result, raise_on_fatal=True)

    @tool
    async def web_fetch(self, params: WebFetchParams) -> ToolOutput:
        result = await run_fetch(
            url=params.url,
            prompt=params.prompt,
            as_of=self._current_as_of(),
            config=self.config,
        )
        return to_tool_output(result, raise_on_fatal=True)


# The environment framework reads ``fn.__doc__`` for each tool's description.
OpenResearcherBackSearch.web_search.__doc__ = SEARCH_DESCRIPTION
OpenResearcherBackSearch.web_fetch.__doc__ = FETCH_DESCRIPTION


class OpenResearcher(Environment):
    """
    OpenResearcher environment: research questions with web search + LLM grading.

    Agent workflow:
    1. Receives a research question requiring web search
    2. Uses web_search tool to find information
    3. Uses web_fetch tool to get detailed content from URLs
    4. Submits answer with explanation, exact_answer, and confidence
    5. Answer is graded by gpt-5-mini comparing to correct answer
    6. Receives reward (1.0 correct, 0.0 incorrect) and feedback
    """

    # web_search / web_fetch come from the SDK's backdated toolset, pinned to
    # OpenReward's backsearch corpus. The cutoff (``web_as_of``) is set per
    # session in ``__init__`` to the UTC date the session was created, and the
    # toolset reads it live on every call.
    toolsets = [OpenResearcherBackSearch]

    def __init__(self, task_spec: JSONObject, secrets: dict[str, str] = {}) -> None:
        """
        Initialize OpenResearcher environment instance.

        Args:
            task_spec: Task specification with qid, question, answer
            secrets: Must contain "openai_api_key" for grading. May contain
                "api_key" (an OpenReward key) for backsearch; otherwise the
                server process's OPENREWARD_API_KEY is used.

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
                "Pass secrets={'openai_api_key': 'sk-...'} when creating session."
            )

        # Backsearch cutoff: the UTC date this session was created. Read live
        # by OpenResearcherBackSearch on every tool call (it outranks the
        # OPENREWARD_WEB_AS_OF env var), so the agent sees the web as it stood
        # on the day it started researching.
        self.web_as_of = today_utc_iso()

        # Read by OpenResearcherBackSearch when it builds its config, so the
        # backsearch key can come from the session rather than the process.
        self.search_secrets = secrets

        # Fail fast if the backdated web service is unconfigured. Without this
        # the first search would raise mid-rollout instead of at session start.
        if WebServiceConfig.from_env(secrets) is None:
            raise ValueError(
                "Backdated web service is not configured: set OPENREWARD_API_KEY "
                "in the server process environment (or pass api_key in secrets)."
            )

        self.openai_client = openai.AsyncClient(api_key=openai_api_key)

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
