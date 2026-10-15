# decision-llm

LoRA fine-tuning of a decision LLM (Qwen3.5-2B) for the **1-token** decision approach: the model
is trained to emit the correct option label as the next token, and at serving time vLLM prefills
the prompt up to the answer and reads the next-token log-probs. No head, no readout tokens, just
causal-LM SFT with a **soft-label** objective (cross-entropy against the teacher's vote-share
distribution, so the model learns calibrated probabilities).

```
decision-llm/
├── configs/qwen35_2b_1tok.yaml    training config
├── src/
│   ├── models/model.py            peft LoRA Qwen3.5-2B (vision tower kept)
│   ├── datasets/catalog_dataset.py catalog → (prompt, soft-target) examples
│   ├── training/trainer.py         soft-label CE training loop
│   └── utils/{render,labels,metrics}.py
└── scripts/train.py                entry point
```

## Quick start

```bash
uv sync                       # install torch(cu130) + transformers + peft + pyarrow + wandb
python -m scripts.train --config configs/qwen35_2b_1tok.yaml
```

Included telegram notifier through `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` env vars. See `configs/qwen35_2b_1tok.yaml` for details on training hyperparameters and LoRA configuration.
