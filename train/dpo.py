import yaml
import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from datasets import load_from_disk

from make_prompts import generate_preference_prompts
from load_model import load_qlora_model_with_lora_adapter
from trl import DPOConfig, DPOTrainer
from peft import PeftModel

with open("config/dpo_config.yaml", "r", encoding="utf-8") as file:
    cfg = yaml.safe_load(file)


if not hasattr(nn.Module, "set_submodule"):

    def set_submodule(self, target, module):
        atoms = target.split(".")
        parent = self
        for name in atoms[:-1]:
            parent = getattr(parent, name)
        setattr(parent, atoms[-1], module)

    nn.Module.set_submodule = set_submodule

# load model and tokenizer
adapter_path = "lora_checkpoints/sft"
MODEL_NAME = cfg["model_name"]

model = load_qlora_model_with_lora_adapter(MODEL_NAME, adapter_path)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

# add think token to tokenizer
tokenizer.add_special_tokens(
    {"additional_special_tokens": ["<|think_start|>", "<|think_end|>"]}
)
model.resize_token_embeddings(len(tokenizer))

# load preference data
train_data = load_from_disk("data/dpo_train_data")
train_ds = generate_preference_prompts(train_data, tokenizer)

# set output directory
stage = cfg["stage_name"]

output_dir = os.path.join("checkpoints", stage)
adapter_output_dir = os.path.join("lora_checkpoints", stage)

# set configuration and trainer for DPO
dpo_config = DPOConfig(
    output_dir=output_dir,
    num_train_epochs=cfg["dpo"]["num_epochs"],
    per_device_train_batch_size=cfg["dpo"]["batch_size"],
    gradient_accumulation_steps=cfg["dpo"]["grad_accum_steps"],
    gradient_checkpointing=False,
    save_strategy="epoch",
    optim="paged_adamw_8bit",
    beta=cfg["dpo"]["beta"],
    learning_rate=float(cfg["dpo"]["learning_rate"]),
    bf16=True,
    lr_scheduler_type="cosine",
    warmup_ratio=cfg["dpo"]["warmup_ratio"],
    do_eval=False,
    group_by_length=False,
    # report_to="wandb",
    # run_name=wandb.run.name,
)

dpo_trainer = DPOTrainer(
    model=model,
    ref_model=None,
    train_dataset=train_ds,
    args=dpo_config,
)

dpo_trainer.train()

# Save LoRA adapter
os.makedirs(adapter_output_dir, exist_ok=True)

assert isinstance(model, PeftModel)

model.save_pretrained(
    adapter_output_dir, safe_serialization=True, save_embedding_layers=False
)
