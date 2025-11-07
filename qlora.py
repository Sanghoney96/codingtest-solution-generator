import torch
import json
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

with open("config.json", "r") as f:
    cfg = json.load(f)

def load_model_and_tokenizer(model_id):
    lora_config = LoraConfig(
                r=cfg["lora"]["rank"],
                lora_alpha=cfg["lora"]["alpha"],
                target_modules=cfg["lora"]["target"],
                lora_dropout=cfg["lora"]["dropout"],
                bias="none",
                task_type="CAUSAL_LM"
            )

    bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

    model = AutoModelForCausalLM.from_pretrained(model_id,
                                                torch_dtype=torch.bfloat16,
                                                device_map="auto",
                                                quantization_config=bnb_config)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    return model, tokenizer