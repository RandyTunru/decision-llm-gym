"""Decision catalog -> training examples.

Reads the assembled catalog snapshot (``states.parquet`` + ``gold.parquet`` + ``question_sets``)
and yields one example per decision: ``{state_text, image_path, question, target, domain, source,
template_id}``. Splits are assigned once: held-out templates are removed entirely, the rest split
into train/val by ``group_key``.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import pyarrow.parquet as pq


def _hash01(*parts: Any) -> float:
    digest = hashlib.sha1("\x1f".join(map(str, parts)).encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16 ** 12)


class DecisionCatalog:
    """Loads and indexes the catalog snapshot + question sets."""

    def __init__(self, catalog_dir: str | Path, snapshot: str, domains: list[str],
                 heldout_templates: list[str], val_fraction: float, seed: int = 0):
        root = Path(catalog_dir)
        states_path = root / "states" / snapshot / "states.parquet"
        gold_path = root / "states" / snapshot / "gold.parquet"
        self.images_dir = root / "images"
        self.domains = set(domains)
        self.heldout = set(heldout_templates)

        self.states = {r["state_id"]: r for r in pq.read_table(states_path).to_pylist()}
        self.gold = [r for r in pq.read_table(gold_path).to_pylist()
                     if r["domain"] in self.domains]

        # question sets: (domain, source) -> {qid: base question, vid: variant question}
        self._qs: Dict[tuple[str, str], Dict[str, Dict[str, dict]]] = {}
        for p in (root / "question_sets").rglob("*.json"):
            doc = json.loads(p.read_text(encoding="utf-8"))
            meta = doc["meta"]
            key = (meta["domain"], meta["source"])
            if meta["domain"] not in self.domains:
                continue
            by_qid, by_vid = {}, {}
            for s in doc.get("question_sets", []):
                for q in s.get("questions", []):
                    by_qid[q["qid"]] = q
                    for v in q.get("variants", []):
                        merged = dict(q)
                        merged.update({k: v[k] for k in ("instructions", "criteria", "type")
                                       if k in v})
                        by_vid[v["vid"]] = merged
            self._qs[key] = {"by_qid": by_qid, "by_vid": by_vid}

        self._splits = self._assign_splits(val_fraction, seed)

    # ---- question resolution -------------------------------------------------

    def resolve_question(self, r: dict) -> Optional[dict]:
        qs = self._qs.get((r["domain"], r["source"]))
        if qs is None:
            return None
        base = qs["by_qid"].get(r["qid"])
        if base is None:
            return None
        if r["kind"] == "base":
            return base
        vid = f"{r['qid']}.{r['kind']}"
        return qs["by_vid"].get(vid, base)

    # ---- splits --------------------------------------------------------------

    def _assign_splits(self, val_fraction: float, seed: int) -> Dict[str, str]:
        """state_id -> 'train' | 'val'; held-out is resolved per decision (template-level)."""
        groups = sorted({self.states[s]["group_key"] for s in self.states})
        val_groups = set(groups[: max(1, int(val_fraction * len(groups)))])
        out = {}
        for sid, st in self.states.items():
            out[sid] = "val" if st["group_key"] in val_groups else "train"
        return out

    # ---- iteration -----------------------------------------------------------

    def examples(self, split: str) -> Iterator[dict]:
        for r in self.gold:
            if r["template_id"] in self.heldout:
                continue
            sid = r["state_id"]
            st = self.states.get(sid)
            if st is None or self._splits.get(sid) != split:
                continue
            q = self.resolve_question(r)
            if q is None:
                continue
            yield {
                "state_text": st.get("text"),
                "image_path": st.get("image_path"),
                "question": q,
                "target": r["target"],
                "domain": r["domain"],
                "source": r["source"],
                "template_id": r["template_id"],
            }
