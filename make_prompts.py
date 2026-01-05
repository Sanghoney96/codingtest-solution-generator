import os
import json
from prompts import REASONING_GENERATION_PROMPT
from datasets import Dataset
from openai import OpenAI

with open("config.json", "r") as f:
    cfg = json.load(f)


def generate_prompts(dataset, tokenizer, is_test=False):
    output_texts = []
    for query, response, reasoning in zip(dataset["query"], dataset["response"], dataset["reasoning"]):
        if is_test == False:
            chunks = response.split("```")
            _, code, explanation = chunks[0], chunks[1], chunks[2]
            
            cot_response = reasoning + "\n\n```" + code + "\n```" + explanation
            
            messages = [
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": cot_response}
                ]
            prompt = tokenizer.apply_chat_template(messages, 
                                                tokenize=False, 
                                                add_generation_prompt=False)
        else:
            system_msg, user_msg = query.split("### Question:", 1)
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": "### Question:" + user_msg}
            ]
            prompt = tokenizer.apply_chat_template(messages, 
                                                tokenize=False, 
                                                add_generation_prompt=True)
            
        output_texts.append(prompt)
        
    output_texts = Dataset.from_dict({"text": output_texts})
    
    return output_texts


_client = None

def get_openai_client():
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client

def generate_reasoning(sample):
    client = get_openai_client()
    query, response = sample['query'], sample['response']
    
    _, question = query.split("### Question:", 1)

    chunks = response.split("```")
    approach, code, explanation = chunks[0], chunks[1], chunks[2]
    
    prompt = REASONING_GENERATION_PROMPT.format(
        question=question,
        approach=approach,
        explanation=explanation,
        code=code
    )
    
    # messages = [
    #     {"role": "user", "content": prompt}
    # ],
    
    # response = client.chat.completions.create(
    #             model=cfg["cot_generation_model"],
    #             messages=messages
    #             )
    
    response = client.responses.create(
                model=cfg["cot_generation_model"],
                input=prompt,
                reasoning={"effort": "high"}
                )
    
    return {"reasoning": response.output_text}