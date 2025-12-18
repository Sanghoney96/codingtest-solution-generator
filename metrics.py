import evaluate

def compute_sacrebleu(prediction, ground_truth):
    sacrebleu = evaluate.load("sacrebleu")
    results_base = sacrebleu.compute(predictions=prediction,
                                    references=ground_truth)
    return