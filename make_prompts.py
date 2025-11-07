def generate_prompts(df, tokenizer):
    output_texts = []
    for query, response in zip(df["query"], df["response"]):
        system_msg, question = query.split("### Question:", 1)

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": question},
            {"role": "assistant", "content": response}
        ]

        prompt = tokenizer.apply_chat_template(messages, 
                                            tokenize=False, 
                                            add_generation_prompt=False)
        output_texts.append(prompt)
        
    return output_texts