import sys
import json
import textwrap
from preprocessing import TextPreprocessor
from pipeline import TextSimplificationPipeline
from evaluate_local import (
    calculate_bleu,
    calculate_sari,
    calculate_rouge,
    calculate_meteor,
)

def load_dev_data(dev_path="dev.json"):
    """Load original text from dev.json."""
    with open(dev_path, "r", encoding="utf-8") as file:
        dev_data = json.load(file)
    return [entry["complex_text"] for entry in dev_data]  # Return original texts

def format_text(text, width=80):
    """Format text for better readability."""
    return textwrap.fill(text, width=width)

def main():
    preprocessor = TextPreprocessor(fasttext_model_path="cc.en.300.bin")  # Initialize preprocessor

    pipeline = TextSimplificationPipeline(  # Initialize pipeline
        model_path="t5_simplification_model",  # Folder with fine-tuned T5 model
        preprocessor=preprocessor,
        freq_threshold=20000
    )

    while True:
        print("\nSelect an option:")
        print("1. Enter text manually for simplification")
        print("2. Run predictions on texts from dev.json")
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

            print("\nResult:")
            print("Original text:")
            print(format_text(user_input))
            print("\nSimplified text:")
            print(format_text(simplified_output))
            print(f"\nBLEU Score: {bleu_score:.2f}")
            print(f"SARI Score: {sari_score:.2f}")
            print(f"ROUGE Scores: {rouge_scores}")
            print(f"METEOR Score: {meteor_score:.2f}")

        elif choice == "2":
            dev_data = load_dev_data("dev.json")
            print(f"\nFound {len(dev_data)} texts in dev.json. Starting predictions...\n")

            for i, text in enumerate(dev_data[:10], 1):  # Limit to 10 examples for quick visualization
                simplified_output = pipeline.simplify(text, apply_postprocess=True)

                # Compute metrics
                bleu_score = calculate_bleu(text, simplified_output)
                sari_score = calculate_sari(text, simplified_output, text)
                rouge_scores = calculate_rouge(text, simplified_output)
                meteor_score = calculate_meteor(text, simplified_output)

                print(f"Text {i}:")
                print("Original text:")
                print(format_text(text))
                print("\nSimplified text:")
                print(format_text(simplified_output))
                print(f"\nBLEU Score: {bleu_score:.2f}")
                print(f"SARI Score: {sari_score:.2f}")
                print(f"ROUGE Scores: {rouge_scores}")
                print(f"METEOR Score: {meteor_score:.2f}")
                print("-" * 80)

        elif choice == "3":
            print("\nGoodbye!")
            break

        else:
            print("\nInvalid option. Please select 1, 2, or 3.")

if __name__ == "__main__":
    main()
