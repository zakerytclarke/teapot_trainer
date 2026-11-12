"""Synthetic arithmetic environment for end-to-end RL smoke tests.

Generates random math word / expression problems with a known numeric answer.
The calculator tool is available so the skeleton can exercise Hugging Face tool
prompting and call parsing for real, even before search / schedule land.
"""

from __future__ import annotations

import operator

from ..env import Env, Task, Trajectory, all_tool_calls, final_assistant_text
from ..rewards import score_numeric_answer

_OPS = [
    ("+", operator.add, "plus"),
    ("-", operator.sub, "minus"),
    ("*", operator.mul, "times"),
]


class SynthMathEnv(Env):
    name = "synth_math"

    def sample_task(self, rng) -> Task:
        difficulty = str(self.options.get("difficulty", "mixed"))
        mode = difficulty
        if difficulty == "mixed":
            mode = rng.choice(["easy", "word"])

        a = int(rng.randint(1, 20))
        b = int(rng.randint(1, 20))
        op_sym, op_fn, op_word = _OPS[int(rng.randrange(len(_OPS)))]
        # Keep division out of v1 so gold stays exact with small ints.
        gold = float(op_fn(a, b))
        expression = f"{a} {op_sym} {b}"

        if mode == "word":
            question = (
                f"What is {a} {op_word} {b}? "
                "You may use the calculator tool. "
                "Reply with a short assistant answer that includes the final number."
            )
        else:
            question = (
                f"Compute {expression}. "
                "You may use the calculator tool if helpful. "
                "Give a brief answer that includes the result."
            )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful math assistant. "
                    "When arithmetic is needed, you may call the calculator tool. "
                    "Then answer the user in natural language and include the final number."
                ),
            },
            {"role": "user", "content": question},
        ]
        return Task(
            task_id=f"math:{mode}:{expression}={gold}",
            messages=messages,
            gold={
                "value": gold,
                "expression": expression,
                "mode": mode,
            },
            meta={"env": self.name},
            allowed_tools=["calculator"],
        )

    def score(self, trajectory: Trajectory) -> tuple[float, dict[str, float]]:
        text = final_assistant_text(trajectory.messages)
        calls = all_tool_calls(trajectory.messages)
        used = any(call.get("name") == "calculator" for call in calls)
        reward, metrics = score_numeric_answer(
            text,
            float(trajectory.task.gold["value"]),
            used_calculator=used,
            require_calculator=bool(self.options.get("require_calculator", False)),
        )
        metrics["num_tool_calls"] = float(len(calls))
        return reward, metrics
