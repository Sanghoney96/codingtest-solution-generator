REASONING_GENERATION_PROMPT = """You are rewriting a model answer 
to create a concise reasoning section for a reference solution.

Your task:
Combine the given APPROACH and EXPLANATION to get a "Reasoning" section,
grounded in the QUESTION and its constraints and problem structure.

Guidelines:
- The reasoning must describe the core algorithmic idea and why it works.
- The reasoning should serve as a high-level algorithmic plan that guides code design, without describing implementation steps.
- Do NOT describe trial-and-error or intermediate thoughts.
- Do NOT mention variable names, loops, or implementation details.
- The reasoning should remain valid even if the code implementation changes.
- You may look at the final CODE only to ensure the algorithm choice is consistent, but do not explain the code itself.

Tone and style:
- Neutral and instructional.
- Focus on *what* algorithm is used, *why it is chosen given the problem structure*, and *why it solves the problem*.
- Avoid redundancy and unnecessary details unrelated to the algorithmic decision or core invariant.

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