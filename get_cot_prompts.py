import json
from datasets import load_dataset, load_from_disk

from make_prompts import generate_reasoning

with open("config.json", "r") as f:
    cfg = json.load(f)
    
train_data = load_dataset(cfg["dataset"], split="train")

dataset_with_reasoning = train_data.map(
    generate_reasoning,
    load_from_cache_file=False
    )

dataset_with_reasoning.save_to_disk("data/train_data_with_reasoning")