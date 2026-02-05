import json
from dotenv import load_dotenv
load_dotenv()

from datasets import load_dataset
from make_prompts import generate_reasoning

with open("config.json", "r") as f:
    cfg = json.load(f)
    
test_data = load_dataset(cfg["dataset"], split="test")
test_data = test_data.filter(
    lambda x: x["response"] is not None and x["response"].strip() != ""
)   # response 필드 값이 비어있는 샘플 제거

test_data_with_reasoning = test_data.map(
    generate_reasoning,
    load_from_cache_file=False
    )

test_data_with_reasoning.save_to_disk("data/test_data_with_reasoning")