import sys
import json
import textwrap
import random
from preprocessing import TextPreprocessor
from pipeline import TextSimplificationPipeline
from evaluate_local import (
    calculate_bleu,
    calculate_sari,
    calculate_rouge,
    calculate_meteor,
    calculate_readability_scores,
)

def load_dev_data(dev_path="dev.json"):
    with open(dev_path, "r", encoding="utf-8") as file:
        dev_data = json.load(file)
    return [entry["complex_text"] for entry in dev_data]  # Return original texts

def format_text(text, width=80):
    return textwrap.fill(text, width=width)

def main():
    preprocessor = TextPreprocessor(fasttext_model_path="cc.en.300.bin")

    pipeline = TextSimplificationPipeline(
        model_path="t5_simplification_model",
        preprocessor=preprocessor,
        freq_threshold=20000
    )

    while True:
        print("\nSelect an option:")
        print("1. Enter text manually for simplification")
        print("2. Run predictions on random texts from dev.json")
        print("3. Exit the application")

        choice = input("Your choice (1/2/3): ").strip()

        if choice == "1":
            user_input = input("\nEnter text for simplification:\n")
            simplified_output = pipeline.simplify(user_input, apply_postprocess=True)

            # Compute metrics
            bleu_score = calculate_bleu(user_input, simplified_output)
            sari_score = calculate_sari(user_input, simplified_output, user_input)
            rouge_scores = calculate_rouge(user_input, simplified_output)
            meteor_score = calculate_meteor(user_input, simplified_output)
            original_readability = calculate_readability_scores(user_input)
            simplified_readability = calculate_readability_scores(simplified_output)

            print("\nResult:")
            print("Original text:")
            print(format_text(user_input))
            print("\nSimplified text:")
            print(format_text(simplified_output))
            print(f"\nBLEU Score: {bleu_score:.2f}")
            print(f"SARI Score: {sari_score:.2f}")
            print(f"ROUGE Scores: {rouge_scores}")
            print(f"METEOR Score: {meteor_score:.2f}")
            print(f"Original Readability: {original_readability}")
            print(f"Simplified Readability: {simplified_readability}")

        elif choice == "2":
            dev_data = load_dev_data("dev.json")
            print(f"\nFound {len(dev_data)} texts in dev.json.")
            random_texts = random.sample(dev_data, min(10, len(dev_data)))
            print("Starting predictions on 10 random texts...\n")

            for i, text in enumerate(random_texts, 1):
                simplified_output = pipeline.simplify(text, apply_postprocess=True)

                # Compute metrics
                bleu_score = calculate_bleu(text, simplified_output)
                sari_score = calculate_sari(text, simplified_output, text)
                rouge_scores = calculate_rouge(text, simplified_output)
                meteor_score = calculate_meteor(text, simplified_output)
                original_readability = calculate_readability_scores(text)
                simplified_readability = calculate_readability_scores(simplified_output)

                print(f"Text {i}:")
                print("Original text:")
                print(format_text(text))
                print("\nSimplified text:")
                print(format_text(simplified_output))
                print(f"\nBLEU Score: {bleu_score:.2f}")
                print(f"SARI Score: {sari_score:.2f}")
                print(f"ROUGE Scores: {rouge_scores}")
                print(f"METEOR Score: {meteor_score:.2f}")
                print(f"Original Readability: {original_readability}")
                print(f"Simplified Readability: {simplified_readability}")
                print("-" * 80)

        elif choice == "3":
            print("\nGoodbye!")
            break

        else:
            print("\nInvalid option. Please select 1, 2, or 3.")

if __name__ == "__main__":
    main()
