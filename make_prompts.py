from prompts import COT_SYSTEM_PROMPT
from datasets import Dataset

def generate_prompts(dataset, tokenizer, is_eval=False):
    output_texts = []
    for query, response in zip(dataset["query"], dataset["response"]):
        system_msg, user_msg = query.split("### Question:", 1)
        cot_user_msg = (
            system_msg + COT_SYSTEM_PROMPT 
            + "### Question:" + user_msg
        )

        if is_eval == False:
            chunks = response.split("```")
            reasoning = chunks[0]
            code = chunks[1] if len(chunks) > 1 else ""
            explanation = chunks[2] if len(chunks) > 2 else ""
            
            assistant_msg = (
                reasoning + "\n\n"
                + explanation + "\n\n"
                + "```" + code + "\n\n```"
            )
            
            messages = [
                    {"role": "user", "content": cot_user_msg},
                    {"role": "assistant", "content": assistant_msg}
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