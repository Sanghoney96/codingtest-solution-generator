import os
import json
import yaml
import wandb

from transformers import AutoTokenizer
from trl import SFTTrainer, SFTConfig
from datasets import load_from_disk

from make_prompts import generate_prompts
from load_model import load_qlora_model
from peft import PeftModel


def train():
    with open("config/sft_config.yaml", "r", encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    # initialize wandb with base configuration
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    wandb.init(
        project=cfg["wandb"]["project"], name=cfg["wandb"]["display_name"], config=cfg
    )

    # # override sweep configuration
    # final_cfg = {}

    # final_cfg["dataset"] = base_cfg["dataset"]
    # final_cfg["model_name"] = base_cfg["model_name"]
    # final_cfg["ckpt_name"] = base_cfg["ckpt_name"]
    # final_cfg["cot_gen_model"] = base_cfg["cot_gen_model"]
    # final_cfg["generation"] = base_cfg["generation"]

    # final_cfg["sft"] = {
    #     "num_epochs": int(
    #         wandb.config.get("num_epochs", base_cfg["sft"]["num_epochs"])
    #     ),
    #     "batch_size": int(
    #         wandb.config.get("batch_size", base_cfg["sft"]["batch_size"])
    #     ),
    #     "grad_accum_steps": int(
    #         wandb.config.get("grad_accum_steps", base_cfg["sft"]["grad_accum_steps"])
    #     ),
    #     "lr": float(
    #         wandb.config.get("learning_rate", base_cfg["sft"]["learning_rate"])
    #     ),
    #     "warmup": float(
    #         wandb.config.get("warmup_ratio", base_cfg["sft"]["warmup_ratio"])
    #     ),
    # }

    # final_cfg["lora"] = {
    #     "rank": int(wandb.config.get("lora_rank", base_cfg["lora"]["rank"])),
    #     "alpha": int(wandb.config.get("lora_alpha", base_cfg["lora"]["alpha"])),
    #     "dropout": float(wandb.config.get("lora_dropout", base_cfg["lora"]["dropout"])),
    #     "target": base_cfg["lora"]["target"],
    # }

    # os.makedirs("sweep_configs", exist_ok=True)
    # with open(f"sweep_configs/config_{wandb.run.id}.json", "w", encoding="utf-8") as f:
    #     json.dump(final_cfg, f, indent=2, ensure_ascii=False)

    # Load dataset
    train_data = load_from_disk("data/train_data_with_reasoning")
    test_data = load_from_disk("data/test_data_with_reasoning")

    # Load model and tokenizer
    model_id = cfg["model_name"]

    model = load_qlora_model(
        model_id=model_id,
        lora_rank=cfg["lora"]["rank"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        target_modules=cfg["lora"]["target"],
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<|think_start|>", "<|think_end|>"]}
    )
    model.resize_token_embeddings(len(tokenizer))

    train_ds = generate_prompts(train_data, tokenizer, is_test=False)
    dev_ds = generate_prompts(test_data, tokenizer, is_test=False)

    # train args
    run_id = wandb.run.id
    output_dir = os.path.join("checkpoints", run_id)
    adapter_output_dir = os.path.join("lora_adapter", cfg["ckpt_name"], run_id)

    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=cfg["sft"]["num_epochs"],
        per_device_train_batch_size=cfg["sft"]["batch_size"],
        gradient_accumulation_steps=cfg["sft"]["grad_accum_steps"],
        save_strategy="no",
        optim="paged_adamw_8bit",
        learning_rate=cfg["sft"]["learning_rate"],
        bf16=True,
        dataset_text_field="text",
        lr_scheduler_type="cosine",
        warmup_ratio=cfg["sft"]["warmup_ratio"],
        do_eval=True,
        eval_strategy="epoch",
        group_by_length=True,
        report_to="wandb",
        run_name=wandb.run.name,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        args=sft_config,
    )

    trainer.train()

    # Save LoRA adapter
    os.makedirs(adapter_output_dir, exist_ok=True)

    assert isinstance(model, PeftModel)

    model.save_pretrained(
        adapter_output_dir, safe_serialization=True, save_embedding_layers=False
    )

    wandb.finish()


if __name__ == "__main__":
    train()
