"""Soft-label LoRA fine-tuning for the 1-token decision approach.

Each decision is one example. The prompt (state + question) is rendered and the assistant reply
prefilled up to the value; the model's next-token distribution over the option labels is compared
to the teacher's soft vote-share distribution with cross-entropy. Text states are batched (one
forward per example, all label continuations together); image states are processed through the
model's processor (image + text) per label.
"""
from __future__ import annotations

import math
import os
import random
import time
from typing import Any, Dict, List, Optional

import torch

from ..utils import labels as label_utils
from ..utils import metrics, render


def _soft_label_ce(model_dists: List[torch.Tensor], teacher_dists: List[list],
                   device: torch.device) -> torch.Tensor:
    losses = []
    for p, t in zip(model_dists, teacher_dists):
        logp = torch.log(p.clamp_min(1e-12))
        losses.append(-(torch.tensor(t, device=device, dtype=torch.float32) * logp).sum())
    return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=device)


class SoftLabelTrainer:
    def __init__(self, model, processor, train_examples: List[dict],
                 val_examples: List[dict], config: Dict[str, Any], device: torch.device,
                 *, images_dir: str, wandb_run=None, start_step: int = 0,
                 optimizer_state: Optional[Dict] = None):
        self.model, self.processor = model, processor
        self.tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
        self.config, self.device = config, device
        self.images_dir, self.wandb = images_dir, wandb_run
        self.step = start_step
        self.accum = int(config.get("gradient_accumulation_steps", 1))
        self.max_steps = int(config["max_steps"])
        self.train_samples, self.val_samples = train_examples, val_examples
        self.batch_size = int(config["batch_size"])
        self.seed = int(config.get("seed", 0))

        params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            params, lr=float(config["learning_rate"]),
            weight_decay=float(config.get("weight_decay", 0.01)), betas=(0.9, 0.95))
        if optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)
        self.best: Dict[str, Any] = {"val/loss": float("inf"), "step": None}
        self.best_state = None

    # ------------------------------------------------------------------ lr

    def get_lr(self, step: int) -> float:
        warmup = int(self.config.get("warmup_steps", 0))
        base = float(self.config["learning_rate"])
        floor = base * float(self.config.get("min_lr_ratio", 0.1))
        if warmup and step < warmup:
            return base * (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, self.max_steps - warmup))
        return floor + (base - floor) * 0.5 * (1 + math.cos(math.pi * progress))

    # ------------------------------------------------------------------ data

    def _image(self, ex: dict):
        if not ex["image_path"]:
            return None
        from PIL import Image

        return Image.open(os.path.join(self.images_dir, ex["image_path"])).convert("RGB")

    # ------------------------------------------------------------------ forward

    def forward(self, examples: List[dict]):
        model_dists: List[torch.Tensor] = []
        teacher: List[list] = []
        for ex in examples:
            teacher.append(label_utils.target_distribution(ex["question"], ex["target"]))
            if ex["image_path"] is None:
                model_dists.append(self._forward_text_one(ex))
            else:
                model_dists.append(self._forward_image_one(ex))
        return model_dists, teacher

    def _continuation_logprob(self, logits: torch.Tensor, cont_ids: List[int]) -> torch.Tensor:
        """Sum of log-softmax over the last len(cont_ids) tokens of one sequence."""
        logp = torch.log_softmax(logits.float(), dim=-1)          # (1, seq_len, V)
        seq_len = logp.shape[1]
        clen = len(cont_ids)
        pred_pos = torch.arange(seq_len - clen - 1, seq_len - 1, device=self.device)
        tgt = torch.tensor(cont_ids, device=self.device)
        return logp[0, pred_pos, tgt].sum()

    def _seq_logprobs(self, seqs: List[List[int]], cont_len: List[int]) -> torch.Tensor:
        """Sum of log-softmax over the last ``cont_len[i]`` tokens of each sequence (one forward)."""
        pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        max_len = max(len(s) for s in seqs)
        K = max(cont_len) + 1
        ids = torch.full((len(seqs), max_len), pad_id, dtype=torch.long, device=self.device)
        att = torch.zeros((len(seqs), max_len), dtype=torch.long, device=self.device)
        for i, s in enumerate(seqs):
            ids[i, max_len - len(s):] = torch.tensor(s, device=self.device)
            att[i, max_len - len(s):] = 1
        logits = self.model(input_ids=ids, attention_mask=att, logits_to_keep=K).logits.float()
        logp = torch.log_softmax(logits, dim=-1)                 # (n_seqs, K, V)
        out = torch.zeros(len(seqs), device=self.device)
        for i, clen in enumerate(cont_len):
            pred_pos = torch.arange(K - 1 - clen, K - 1, device=self.device)
            tgt = torch.tensor(seqs[i][-clen:], device=self.device)
            out[i] = logp[i, pred_pos, tgt].sum()
        return out

    def _forward_text_one(self, ex: dict) -> torch.Tensor:
        """Chunked soft-label forward: the state is truncated and labels batched in small groups."""
        state = ex["state_text"] or ""
        max_chars = int(self.config.get("max_state_chars", 0))
        if max_chars:
            state = state[:max_chars]
        msgs = render.messages(state, [ex["question"]])
        labels, p_ids, c_ids = label_utils.option_inputs(self.tok, msgs, ex["question"])
        chunk = max(1, int(self.config.get("max_labels_per_forward", 8)))
        parts = []
        for start in range(0, len(c_ids), chunk):
            group = c_ids[start:start + chunk]
            parts.append(self._seq_logprobs([p_ids + c for c in group], [len(c) for c in group]))
        seq_logprob = torch.cat(parts)
        z = torch.logsumexp(seq_logprob, dim=0)
        return (seq_logprob - z).exp()

    def _forward_image_one(self, ex: dict) -> torch.Tensor:
        """One processor pass per label of an image example (image + text together)."""
        image = self._image(ex)
        vals = label_utils.label_values(ex["question"])
        msgs = render.messages(ex["state_text"], [ex["question"]], image=image)
        prefix_text = (self.processor.apply_chat_template(msgs, tokenize=False,
                                                          add_generation_prompt=True)
                       + render.prefill(ex["question"]))
        logps = []
        for v in vals:
            enc = self.processor(text=[prefix_text + v + "}}"], images=[image],
                                 return_tensors="pt")
            enc = {k: t.to(self.device) for k, t in enc.items()
                   if k in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")}
            logits = self.model(**enc).logits
            cont_ids = self.tok(v + "}}", add_special_tokens=False)["input_ids"]
            logps.append(self._continuation_logprob(logits, cont_ids))
        stacked = torch.stack(logps)
        z = torch.logsumexp(stacked, dim=0)
        return (stacked - z).exp()

    # ------------------------------------------------------------------ loop

    def _batches(self, split: str):
        examples = self.train_samples if split == "train" else self.val_samples
        rng = random.Random(self.seed + self.step)
        order = list(range(len(examples)))
        if split == "train":
            rng.shuffle(order)
        for start in range(0, len(order), self.batch_size):
            yield [examples[i] for i in order[start:start + self.batch_size]]

    def train(self):
        cfg = self.config
        log_every = int(cfg.get("log_every", 10))
        eval_every = int(cfg.get("eval_every", 250))
        save_every = int(cfg.get("save_every", 500))
        window = {"loss": 0.0, "n": 0, "t0": time.time()}
        self.model.train()  # gradient checkpointing only engages in train mode

        while self.step < self.max_steps:
            for g in self.optimizer.param_groups:
                g["lr"] = self.get_lr(self.step)
            self.optimizer.zero_grad(set_to_none=True)
            for _ in range(self.accum):
                batch = next(self._batches("train"))
                model_dists, teacher = self.forward(batch)
                loss = _soft_label_ce(model_dists, teacher, self.device) / self.accum
                loss.backward()
                window["loss"] += float(loss) * self.accum
                window["n"] += 1
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(cfg.get("grad_clip", 1.0)))
            self.optimizer.step()
            self.step += 1

            if self.step % log_every == 0 or self.step in (1, self.max_steps):
                n, dt = max(1, window["n"]), max(1e-6, time.time() - window["t0"])
                print(f"step {self.step}/{self.max_steps}  loss {window['loss'] / n:.4f}  "
                      f"lr {self.get_lr(self.step):.2e}", flush=True)
                if self.wandb:
                    self.wandb.log({"train/loss": window["loss"] / n,
                                    "train/lr": self.get_lr(self.step)}, step=self.step)
                window = {"loss": 0.0, "n": 0, "t0": time.time()}

            if self.step % eval_every == 0 or self.step == self.max_steps:
                metrics_dict = self.evaluate()
                if self.wandb:
                    self.wandb.log(metrics_dict, step=self.step)
                print(f"  eval step {self.step}: " +
                      "  ".join(f"{k} {v:.4f}" for k, v in metrics_dict.items()), flush=True)
                if metrics_dict["val/loss"] < self.best["val/loss"]:
                    self.best = dict(metrics_dict, step=self.step)
                    self.best_state = {k: v.detach().cpu().clone()
                                       for k, v in self.model.state_dict().items()}

            if save_every and self.step % save_every == 0:
                self.save_checkpoint(self.step)

        self.save_checkpoint("final")
        return self.step, self.best

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        self.model.eval()
        all_model, all_teacher = [], []
        for batch in self._batches("val"):
            model_dists, teacher = self.forward(batch)
            for p, t in zip(model_dists, teacher):
                all_model.append([float(x) for x in p])
                all_teacher.append(t)
            if len(all_model) >= int(self.config.get("eval_max_batches", 100)) * self.batch_size:
                break
        self.model.train()
        return {
            "val/loss": metrics.soft_cross_entropy(all_model, all_teacher),
            "val/acc": metrics.argmax_accuracy(
                [int(torch.tensor(m).argmax()) for m in all_model],
                [int(torch.tensor(t).argmax()) for t in all_teacher]),
        }

    # ------------------------------------------------------------------ checkpoint

    def save_checkpoint(self, label: Any) -> str:
        cfg = self.config
        path = os.path.join(cfg["checkpoint_dir"], f"model_step_{label}.pt")
        os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
        trainable = {k: v for k, v in self.model.state_dict().items() if "lora" in k}
        torch.save({"model_state_dict": trainable, "optimizer_state_dict": self.optimizer.state_dict(),
                    "step": self.step, "config": cfg}, path)
        keep = int(cfg.get("keep_checkpoints", 0))
        if keep and isinstance(label, int):
            steps = sorted(int(f[len("model_step_"):-3]) for f in os.listdir(cfg["checkpoint_dir"])
                           if f.startswith("model_step_") and f[len("model_step_"):-3].isdigit())
            for old in steps[:-keep]:
                os.remove(os.path.join(cfg["checkpoint_dir"], f"model_step_{old}.pt"))
        return path

    def export_adapter(self) -> str:
        """Save the LoRA adapter for vLLM (adapter_config.json + adapter_model.safetensors)."""
        cfg = self.config
        out = cfg["adapter_dir"]
        os.makedirs(out, exist_ok=True)
        self.model.save_pretrained(out)
        self.processor.tokenizer.save_pretrained(out)
        return out
