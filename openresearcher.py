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


from openreward.environments import Environment, JSONObject, TextBlock, ToolOutput, tool
from openreward.toolsets import WebToolset

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

    # web_search / web_fetch come from the SDK rather than being hand-rolled here.
    # Which provider answers is process configuration (OPENREWARD_SEARCH_BACKEND,
    # default "backsearch"), so changing search provider needs no change here.
    #
    # The toolset owns the error split too: an unfetchable page stays tool output
    # the agent can act on, while a missing key or exhausted quota raises so the
    # rollout ends with a blank reward rather than a score that reads as a bad answer.
    toolsets = [WebToolset]

    # Search hits keep their snippets, as the prompt promises. Off in the SDK by
    # default, which would force a fetch per candidate just to triage results.
    web_include_snippets = True

    def __init__(self, task_spec: JSONObject, secrets: dict[str, str] = {}) -> None:
        """
        Initialize OpenResearcher environment instance.

        Args:
            task_spec: Task specification with qid, question, answer
            secrets: Must contain "openai_api_key" for grading; search credentials
                (api_key / tavily_api_key) are forwarded to the search backend

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

        # Read live by WebToolset on every tool call, so the search backend takes its
        # credentials from the session rather than the server process. The configured
        # backend picks the key it needs: `api_key` for backsearch, `tavily_api_key`
        # for tavily. No up-front check — which key is required depends on the backend.
        self.search_secrets = secrets

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
