#!/usr/bin/env python3
"""LoRA fine-tune a decision LLM on the decision catalog (1-token soft-label objective).

    python -m scripts.train --config configs/qwen35_2b_1tok.yaml
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")  # HF_TOKEN, WANDB_*, Telegram

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import yaml  # noqa: E402


def _notify(msg: str) -> None:
    import json
    import urllib.request

    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat = os.environ.get("TELEGRAM_CHAT_ID")
    if not (token and chat):
        return
    try:
        body = json.dumps({"chat_id": chat, "text": msg}).encode()
        urllib.request.urlopen(
            f"https://api.telegram.org/bot{token}/sendMessage", body,
            headers={"Content-Type": "application/json"}, timeout=10)
    except Exception:  # noqa: BLE001  # never let a notification crash the run
        pass


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default="configs/qwen35_2b_1tok.yaml")
    args = ap.parse_args(argv)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # --- data ---
    from src.datasets.catalog_dataset import DecisionCatalog

    cat = DecisionCatalog(
        cfg["catalog_dir"], cfg["snapshot"], cfg["domains"],
        cfg.get("heldout_templates", []), float(cfg.get("val_fraction", 0.1)),
        seed=int(cfg.get("seed", 0)))
    train_examples = list(cat.examples("train"))
    val_examples = list(cat.examples("val"))
    print(f"examples: {len(train_examples)} train, {len(val_examples)} val "
          f"(held-out templates: {len(cfg.get('heldout_templates', []))})")

    images_dir = str(Path(cfg["catalog_dir"]) / "images")

    # --- model ---
    from src.models.model import build_model

    model, processor = build_model(cfg)
    model.to(device)
    model.config.use_cache = False

    # --- trainer ---
    from src.training.trainer import SoftLabelTrainer

    wandb_run = None
    if os.environ.get("WANDB_API_KEY") or os.environ.get("WANDB_MODE") == "offline":
        import wandb
        wandb_run = wandb.init(project=cfg["project_name"], name=cfg["run_name"],
                               config=cfg, dir=cfg["wandb_dir"],
                               id=cfg.get("run_id") or None, resume="allow")

    trainer = SoftLabelTrainer(model, processor, train_examples, val_examples, cfg, device,
                               images_dir=images_dir, wandb_run=wandb_run)

    t0 = time.time()
    try:
        final_step, best = trainer.train()
    finally:
        adapter = trainer.export_adapter()

    if wandb_run:
        wandb_run.finish()

    print(f"done in {time.time() - t0:.0f}s; best {best['step']} val/loss {best['val/loss']:.4f}; "
          f"adapter -> {adapter}")
    _notify(f"decision-llm {cfg['run_name']} finished in {time.time() - t0:.0f}s "
            f"(best val/loss {best['val/loss']:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
