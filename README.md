# decision-llm

LoRA fine-tuning of a decision LLM (Qwen3.5-2B) for the **1-token** decision approach: the model
is trained to emit the correct option label as the next token, and at serving time vLLM prefills
the prompt up to the answer and reads the next-token log-probs. No head, no readout tokens — just
causal-LM SFT with a **soft-label** objective (cross-entropy against the teacher's vote-share
distribution, so the model learns calibrated probabilities).

The dataset comes from the sibling `../catalog` (the decision catalog). This repo only *reads* the
catalog and converts it to training examples; it does not own or modify the catalog.

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

`python-dotenv` loads `../decision-llm/.env` (HF_TOKEN, WANDB_*, Telegram) at the top of
`scripts/train.py`, mirroring the i-jepa-clone layout.
