"""
Test agent for OpenResearcher environment.

Runs a real model (gpt-5.2 via the Responses API) against the environment: the
agent researches each question through the backdated web_search / web_fetch
tools (OpenReward's backsearch corpus, cutoff = the day the session starts) and
calls submit_answer. Every search hit is printed with the corpus that served it,
so you can see what kind of URLs the agent is working from.

Runs against a local `python server.py` (localhost:8080) by default. The local
server process must have OPENREWARD_API_KEY exported, or pass it as the
`api_key` secret (this script does both when the variable is set). Set
DEPLOYED=1 to hit the deployed environment instead.

    export OPENAI_API_KEY=sk-...           # grader + policy
    export OPENREWARD_API_KEY=or_...        # backsearch
    python server.py &                      # in another shell
    NUM_TASKS=2 python test_agent.py
    DEPLOYED=1 NUM_TASKS=1 python test_agent.py
"""

import asyncio
import json
import os
from collections import Counter

from openai import AsyncOpenAI
from openreward import AsyncOpenReward


async def run_task(environment, oai_client, task, tools, *, model, secrets, max_turns, corpus_tally):
    finished = False
    async with environment.session(task=task, secrets=secrets) as session:
        prompt = await session.get_prompt()
        prompt_text = prompt[0].text
        print(f"\nPrompt preview: {prompt_text[:200].replace(chr(10), ' ')}...")

        input_list = [{"role": "user", "content": prompt_text}]
        turn = 0
        reward = None

        while not finished and turn < max_turns:
            turn += 1
            response = await oai_client.responses.create(
                model=model,
                tools=tools,
                input=input_list,
            )
            input_list += response.output

            for item in response.output:
                if item.type == "function_call":
                    args = json.loads(str(item.arguments))
                    tool_result = await session.call_tool(item.name, args)
                    reward = tool_result.reward
                    finished = tool_result.finished
                    text = tool_result.blocks[0].text if tool_result.blocks else ""
                    meta = tool_result.metadata or {}

                    if item.name == "web_search":
                        hits = meta.get("hits") or []
                        corpus_tally.update(h.get("corpus") for h in hits)
                        print(f"\n[turn {turn}] web_search: {args.get('query', '')!r} -> {len(hits)} hits"
                              + (f" (error={meta['error']})" if meta.get("error") else ""))
                        for h in hits:
                            print(f"    [{h.get('corpus') or '?':<11}] {str(h.get('publish_date') or '')[:10]:<10} {h.get('url')}")
                    elif item.name == "web_fetch":
                        print(f"\n[turn {turn}] web_fetch: {args.get('url', '')} -> {len(text)} chars"
                              + (f" (error={meta['error']})" if meta.get("error") else ""))
                    elif item.name == "submit_answer":
                        print(f"\n[turn {turn}] submit_answer: {args.get('exact_answer', '')!r} (confidence {args.get('confidence')})")
                        print(f"    reward={reward} | expected={meta.get('correct_answer', '')!r}")

                    input_list.append({
                        "type": "function_call_output",
                        "call_id": item.call_id,
                        "output": text,
                    })
                    if finished:
                        break

                elif item.type == "message":
                    for block in getattr(item, "content", None) or []:
                        if getattr(block, "type", "") == "output_text" and block.text:
                            print(f"\n[turn {turn}] model: {block.text[:160].replace(chr(10), ' ')}...")

            if not any(i.type == "function_call" for i in response.output):
                print("Model produced no tool call; stopping this task.")
                break

        if turn >= max_turns and not finished:
            print(f"Reached max turns ({max_turns}) without submit_answer.")
        return reward, finished, turn


async def main():
    MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-5.2")
    SPLIT = "train"
    NUM_TASKS = int(os.environ.get("NUM_TASKS", "1"))
    TASK_OFFSET = int(os.environ.get("TASK_OFFSET", "0"))
    MAX_TURNS = int(os.environ.get("MAX_TURNS", "20"))
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENREWARD_API_KEY = os.getenv("OPENREWARD_API_KEY")

    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable required (grader + policy model).")

    deployed = bool(os.environ.get("DEPLOYED"))
    ENV_NAME = os.environ.get("ENV_NAME", "GeneralReasoning/OpenResearcher" if deployed else "openresearcher")
    base_url = None if deployed else "http://localhost:8080"

    or_client = AsyncOpenReward()
    oai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    print(f"Environment: {ENV_NAME} ({base_url or 'deployed'})")
    environment = or_client.environments.get(name=ENV_NAME, base_url=base_url)

    tasks = await environment.list_tasks(split=SPLIT)
    tools = await environment.list_tools(format="openai")
    print(f"Found {len(tasks)} tasks in split '{SPLIT}'")
    print(f"Tools: {[t['name'] for t in tools]}")

    secrets = {
        "openai_api_key": OPENAI_API_KEY,
        **({"api_key": OPENREWARD_API_KEY} if OPENREWARD_API_KEY else {}),
    }

    corpus_tally: Counter = Counter()
    results = []
    for task in tasks[TASK_OFFSET:TASK_OFFSET + NUM_TASKS]:
        print(f"\n{'=' * 80}\nTask qid={task.task_spec.get('qid')}\n{'=' * 80}")
        reward, finished, turns = await run_task(
            environment, oai_client, task, tools,
            model=MODEL_NAME, secrets=secrets, max_turns=MAX_TURNS, corpus_tally=corpus_tally,
        )
        results.append((task.task_spec.get("qid"), reward, finished, turns))

    print(f"\n{'=' * 80}\nSummary\n{'=' * 80}")
    for qid, reward, finished, turns in results:
        print(f"qid={qid}: reward={reward} finished={finished} turns={turns}")
    print(f"Search hits by corpus across the run: {dict(corpus_tally)}")


if __name__ == "__main__":
    asyncio.run(main())
