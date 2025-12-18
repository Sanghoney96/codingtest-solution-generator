from prompts import COT_SYSTEM_PROMPT

def generate_prompts(dataset, tokenizer, is_eval=False):
    output_texts = []
    for query, response in zip(dataset["query"], dataset["response"]):
        system_msg, user_msg = query.split("### Question:", 1)
        cot_user_msg = (
            system_msg + COT_SYSTEM_PROMPT 
            + "### Question:" + user_msg
        )
        
        chunks = response.split("```")
        reasoning = chunks[0]
        code = chunks[1] if len(chunks) > 1 else ""
        explanation = chunks[2] if len(chunks) > 2 else ""
        
        assistant_msg = (
            reasoning + "\n\n"
            + explanation + "\n\n"
            + "```" + code + "\n\n```"
        )

        if is_eval == False:
            messages = [
                    {"role": "user", "content": cot_user_msg},
                    {"role": "assistant", "content": assistant_msg}
                ]
            prompt = tokenizer.apply_chat_template(messages, 
                                                tokenize=False, 
                                                add_generation_prompt=False)
        else:
            messages = [
                    {"role": "user", "content": cot_user_msg}
                ]
            prompt = tokenizer.apply_chat_template(messages, 
                                                tokenize=False, 
                                                add_generation_prompt=True)
            
        output_texts.append(prompt)
        
    return output_texts


def preprocess_inputs(df):
    system_msg, user_msg = df['query'].split("### Question:", 1)
    cot_prompt = system_msg + COT_SYSTEM_PROMPT + "### Question: " + user_msg
    
    chunks = df["response"].split("```")
    reasoning = chunks[0]
    code = chunks[1] if len(chunks) > 1 else ""
    explanation = chunks[2] if len(chunks) > 2 else ""
    
    completion = (
        reasoning
        + explanation + "\n\n"
        + "```" + code + "\n```"
    )
    
    return {
        "prompt": [{"role": "user", "content": cot_prompt}],
        "completion": [
            {"role": "assistant", "content": completion}
        ],
    }