import evaluate
from textstat import flesch_reading_ease, flesch_kincaid_grade

def calculate_bleu(original, simplified):
    """Calculate BLEU score."""
    sacrebleu = evaluate.load("sacrebleu")
    references = [[original]]  # Nested list for references
    predictions = [simplified]
    result = sacrebleu.compute(predictions=predictions, references=references)
    return result["score"]

def calculate_sari(original, simplified, references):
    """Calculate SARI score."""
    sari_metric = evaluate.load("sari")
    references = [[references]]  # Nested list for references
    result = sari_metric.compute(
        sources=[original],
        predictions=[simplified],
        references=references
    )
    if "sari" in result:
        return result["sari"]
    elif "SARI" in result:
        return result["SARI"]
    else:
        raise KeyError("SARI score not found in the result structure.")

def calculate_rouge(original, simplified):
    """Calculate ROUGE scores."""
    rouge = evaluate.load("rouge")
    result = rouge.compute(
        predictions=[simplified],
        references=[original]
    )
    return result

def calculate_meteor(original, simplified):
    """Calculate METEOR score."""
    meteor = evaluate.load("meteor")
    result = meteor.compute(
        predictions=[simplified],
        references=[original]
    )
    return result["meteor"]

def calculate_readability_scores(text):
    """Calculate readability scores using Flesch-Kincaid metrics."""
    flesch_score = flesch_reading_ease(text)
    fk_grade = flesch_kincaid_grade(text)
    return {
        "flesch_reading_ease": flesch_score,
        "flesch_kincaid_grade": fk_grade
    }
