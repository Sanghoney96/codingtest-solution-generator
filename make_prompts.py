import os
from prompts import COT_SYSTEM_PROMPT, REASONING_GENERATION_PROMPT
from datasets import Dataset
from openai import OpenAI


def generate_prompts(dataset, tokenizer, is_eval=False):
    output_texts = []
    for query, response in zip(dataset["query"], dataset["response"]):
        system_msg, user_msg = query.split("### Question:", 1)
        cot_user_msg = (
            system_msg + COT_SYSTEM_PROMPT 
            + "### Question:" + user_msg
        )

        if is_eval == False:
            messages = [
                    {"role": "user", "content": cot_user_msg},
                    {"role": "assistant", "content": response}
                ]
            prompt = tokenizer.apply_chat_template(messages, 
                                                tokenize=False, 
                                                add_generation_prompt=False)
        else:
            messages = [
                {"role": "system", "content": system_msg + COT_SYSTEM_PROMPT},
                {"role": "user", "content": "### Question:" + user_msg}
            ]
            prompt = tokenizer.apply_chat_template(messages, 
                                                tokenize=False, 
                                                add_generation_prompt=True)
            
        output_texts.append(prompt)
        
    output_texts = Dataset.from_dict({"text": output_texts})
    
    return output_texts


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_reasoning(sample):
    query, response = sample['query'], sample['response']
    
    _, question = query.split("### Question:", 1)

    chunks = response.split("```")
    approach = chunks[0]
    code = chunks[1] if len(chunks) > 1 else ""
    explanation = chunks[2] if len(chunks) > 2 else ""
    
    prompt = REASONING_GENERATION_PROMPT.format(
        question=question,
        approach=approach,
        explanation=explanation,
        code=code
    )
    
    messages = [
                {"role": "user", "content": prompt}
            ]

    response = client.chat.completions.create(
                model="gpt-5-mini",
                messages=messages
            )
    
    return {"reasoning": response.choices[0].message.content}