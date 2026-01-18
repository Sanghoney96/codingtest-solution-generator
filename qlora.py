import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_qlora_model(model_id, lora_rank, lora_alpha, lora_dropout, target_modules):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=bnb_config,
        attn_implementation="flash_attention_2",
    )

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=int(lora_rank),
        lora_alpha=int(lora_alpha),
        target_modules=target_modules,
        lora_dropout=float(lora_dropout),
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    return model


def load_trained_model(model_id, adaptor_path):
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )

    model = PeftModel.from_pretrained(base_model, adaptor_path)

    model.config.use_cache = True
    model.eval()

    return model
