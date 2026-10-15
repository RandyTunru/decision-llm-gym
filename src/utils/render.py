"""Prompt rendering for the 1-token decision approach.

Byte-identical to the benchmark's ``source/jbench/render.py`` so training and serving agree
exactly on the prompt. A question is rendered as a numbered task with its option list, followed
by the JSON answer format. Text states go in the user message as plain text; image states use the
transformers multimodal content format (image part + text part).
"""
from __future__ import annotations

import json
from typing import Any, Optional

SYSTEM_PROMPT = ("You are a decision engine. Read the state and answer each task. "
                 "Return JSON only, no explanation.")

IMAGE_STATE_TEXT = "State: the attached image."


def canonical_state_text(state) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, indent=2)


def render_task(i: int, q: dict) -> str:
    lines = [f"## Task {i} ({q['qid']}): {q['instructions']}"]
    t = q["type"]
    if t == "choice":
        lines.append("- type: choice, answer with exactly one of the option keys")
        lines += [f"- {k}: {v}" for k, v in q["criteria"].items()]
    elif t == "score":
        lines.append("- type: score, answer with an integer level")
        lines += [f"- level {lvl}: {v}" for lvl, v in enumerate(q["criteria"])]
    elif t == "noul":
        lines.append("- type: noul, answer with true or false")
    else:
        raise ValueError(f"unknown primitive {t!r}")
    return "\n".join(lines)


def answer_format(questions: list[dict]) -> str:
    cells = []
    for q in questions:
        t = q["type"]
        if t == "noul":
            v = "true|false"
        elif t == "choice":
            v = "|".join(json.dumps(k, ensure_ascii=False) for k in q["criteria"])
        else:
            v = f"<0..{len(q['criteria']) - 1}>"
        cells.append(f'{json.dumps(q["qid"])}: {{"{t}": {v}}}')
    return "{" + ", ".join(cells) + "}"


def render_prompt(state_text: str, questions: list[dict]) -> str:
    tasks = "\n\n".join(render_task(i + 1, q) for i, q in enumerate(questions))
    return f"{state_text}\n\n{tasks}\n\nAnswer format (JSON): {answer_format(questions)}"


def messages(state_text: Optional[str], questions: list[dict],
             image: Any = None) -> list[dict]:
    """Chat messages for one state + one question.

    ``image`` is a PIL image / path / bytes for image states (passed through to the model's
    processor); ``state_text`` carries the goal/controls text (may be None for pure images).
    """
    if image is not None:
        content = [{"type": "image", "image": image}]
        if state_text:
            content.append({"type": "text",
                            "text": render_prompt(state_text, questions)})
        return [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content}]
    text = render_prompt(canonical_state_text(state_text or ""), questions)
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text}]


def prefill(q: dict) -> str:
    """The assistant-reply prefill up to the value: ``{"<qid>": {"<type>": ``.

    This is the compact layout; the serving 1-token runner learns the model's own layout via a
    probe and prefills to the same value position, so only the label-relative probabilities (the
    softmax over label continuations) need to transfer from training to serving.
    """
    return "{" + json.dumps(q["qid"]) + ': {"' + q["type"] + '": '
