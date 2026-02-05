import yaml
from datasets import Dataset


def generate_prompts(dataset, tokenizer, is_test=False):
    output_texts = []
    for query, response, reasoning in zip(
        dataset["query"], dataset["response"], dataset["reasoning"]
    ):
        if is_test == False:
            chunks = response.split("```")
            _, code, explanation = chunks[0], chunks[1], chunks[2]

            cot_response = (
                "<|think_start|>\n"
                + reasoning
                + "\n<|think_end|>"
                + "\n\n```"
                + code
                + "\n```"
                + explanation
            )

            messages = [
                {"role": "user", "content": query},
                {"role": "assistant", "content": cot_response},
            ]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        else:
            messages = [{"role": "user", "content": query}]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

        output_texts.append(prompt)

    output_texts = Dataset.from_dict({"text": output_texts})

    return output_texts


def generate_preference_prompts(dataset, tokenizer):
    prompts = []
    chosen_li = []
    rejected_li = []

    for query, response, reasoning, rejected in zip(
        dataset["query"], dataset["response"], dataset["reasoning"], dataset["rejected"]
    ):
        query_messages = [{"role": "user", "content": query}]
        prompt = tokenizer.apply_chat_template(
            query_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        chunks = response.split("```")
        _, code, explanation = chunks[0], chunks[1], chunks[2]

        chosen = (
            "<|think_start|>\n"
            + reasoning
            + "\n<|think_end|>"
            + "\n\n```"
            + code
            + "\n```"
            + explanation
            + "<|im_end|>"
        )

        prompts.append(prompt)
        chosen_li.append(chosen)
        rejected_li.append(rejected + "<|im_end|>")

    return Dataset.from_dict(
        {
            "prompt": prompts,
            "chosen": chosen_li,
            "rejected": rejected_li,
        }
    )
