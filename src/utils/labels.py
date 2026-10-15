"""Labels, label values and the shared-prefix / per-label-continuation token split.

The 1-token approach reads the answer at the next token after the prefill. For a soft-label
objective we need, per question, the model's log-prob of each label's full continuation, computed
from one forward pass over the shared prompt prefix plus each label's continuation (the same trick
as the benchmark's ``_option_probs``).
"""
from __future__ import annotations

import json
from typing import Any, Optional


def labels_for(q: dict) -> list[str]:
    """Answer labels in authored order."""
    if q["type"] == "noul":
        return ["false", "true"]
    if q["type"] == "choice":
        return list(q["criteria"].keys())
    return [str(i) for i in range(len(q["criteria"]))]


def label_values(q: dict) -> list[str]:
    """The JSON value text of each label, in ``labels_for`` order.

    choice -> ``"key"`` (quoted), noul -> ``true``/``false``, score -> ``"0".."k-1"``.
    """
    vals = []
    for k in labels_for(q):
        vals.append(json.dumps(k, ensure_ascii=False) if q["type"] == "choice" else k)
    return vals


def option_inputs(tok, messages: list[dict], q: dict) -> tuple[list[str], list[int], list[list[int]]]:
    """(labels, shared prefix ids, continuation ids per label) for a question.

    The prompt is the chat template applied to ``messages`` plus the prefill, then each label's
    value + closing ``}}``. Returns the longest common token prefix (shared, scored once) and the
    per-label continuation token ids. Mirror of the benchmark's ``_option_inputs``.
    """
    from .render import prefill

    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prefix = prompt + prefill(q)
    vals = label_values(q)
    full = [tok(prefix + v + "}}", add_special_tokens=False)["input_ids"] for v in vals]
    n = min(map(len, full))
    lcp = next((i for i in range(n) if len({f[i] for f in full}) > 1), n)
    lcp = min(lcp, n - 1)  # every label keeps at least one scored token
    labels = labels_for(q)
    return labels, full[0][:lcp], [f[lcp:] for f in full]


def target_distribution(q: dict, target: list[float]) -> list[float]:
    """Normalise the catalog's soft ``target`` over ``labels_for(q)`` order (already aligned)."""
    labels = labels_for(q)
    if len(target) != len(labels):
        # fallback: renormalise whatever length we were given onto the label count
        t = list(target) + [0.0] * (len(labels) - len(target))
    else:
        t = list(target)
    z = sum(t) or 1.0
    return [x / z for x in t]
