COT_SYSTEM_PROMPT = """Before solving a problem, think step by step and explain your reasoning. 
Start with your approach and explanation, then write the final code.\n\n"""

REASONING_GENERATION_PROMPT = """You are rewriting a model answer 
to create a concise reasoning section for a reference solution.

Your task:
Combine the given APPROACH and EXPLANATION to get "Reasoning" section.

Guidelines:
- The reasoning must describe the core algorithmic idea and why it works.
- Use at most 1–4 sentences.
- Do NOT provide step-by-step reasoning.
- Do NOT describe trial-and-error or intermediate thoughts.
- Do NOT mention variable names, loops, or implementation details.
- The reasoning should remain valid even if the code implementation changes.
- You may look at the final CODE only to ensure the algorithm choice is consistent, but do not explain the code itself.

Tone and style:
- Neutral and instructional.
- Focus on *what* algorithm is used and *why* it solves the problem.
- Avoid redundancy and unnecessary details.

Input:
[QUESTION]
{question}

[ORIGINAL APPROACH]
{approach}

[ORIGINAL EXPLANATION]
{explanation}

[FINAL CODE]  (for consistency check only)
{code}

Output format:
Return only the rewritten reasoning text.
Do NOT include the code.
Do NOT include headings, bullet points, or extra commentary.
"""