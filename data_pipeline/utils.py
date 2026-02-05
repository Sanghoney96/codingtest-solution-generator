import os
import yaml
from openai import OpenAI
from prompts import REASONING_GENERATION_PROMPT


with open("config/sft_config.yaml") as file:
    cfg = yaml.load(file, Loader=yaml.FullLoader)


_client = None


def get_openai_client():
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


def generate_reasoning(sample):
    client = get_openai_client()
    query, response = sample["query"], sample["response"]

    _, question = query.split("### Question:", 1)

    chunks = response.split("```")
    approach, code, explanation = chunks[0], chunks[1], chunks[2]

    prompt = REASONING_GENERATION_PROMPT.format(
        question=question, approach=approach, explanation=explanation, code=code
    )

    response = client.responses.create(
        model=cfg["cot_gen_model"], input=prompt, reasoning={"effort": "high"}
    )

    return {"reasoning": response.output_text}
