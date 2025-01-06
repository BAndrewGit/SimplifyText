import sys
import json
import textwrap
from preprocessing import TextPreprocessor
from pipeline import TextSimplificationPipeline
import evaluate  # Biblioteca pentru metrici

def load_dev_data(dev_path="dev.json"):
    """Încărcați textul original din dev.json."""
    with open(dev_path, "r", encoding="utf-8") as file:
        dev_data = json.load(file)
    return [entry["complex_text"] for entry in dev_data]  # Returnează textele originale

def format_text(text, width=80):
    """Formatează textul pentru afișare în linii scurte."""
    return textwrap.fill(text, width=width)

def calculate_bleu(original, simplified):
    """Calculăm scorul BLEU pentru textul simplificat."""
    sacrebleu = evaluate.load("sacrebleu")
    references = [[original]]  # sacrebleu așteaptă liste de referințe
    predictions = [simplified]
    result = sacrebleu.compute(predictions=predictions, references=references)
    return result["score"]

def main():
    preprocessor = TextPreprocessor(fasttext_model_path="cc.en.300.bin")  # Inițializăm preprocesorul

    pipeline = TextSimplificationPipeline(  # Inițializăm pipeline-ul
        model_path="t5_simplification_model",  # Folderul cu modelul T5 antrenat
        preprocessor=preprocessor,
        freq_threshold=20000
    )

    while True:
        print("\nSelectează opțiunea:")
        print("1. Introdu manual textul pentru simplificare")
        print("2. Rulează predictii pe textele din dev.json")
        print("3. Ieși din aplicație")

        choice = input("Opțiunea ta (1/2/3): ").strip()

        if choice == "1":
            user_input = input("\nIntrodu textul pentru simplificare:\n")
            simplified_output = pipeline.simplify(user_input, apply_postprocess=True)
            bleu_score = calculate_bleu(user_input, simplified_output)

            print("\nRezultat:")
            print("Original text:")
            print(format_text(user_input))
            print("\nSimplified text:")
            print(format_text(simplified_output))
            print(f"\nBLEU Score: {bleu_score:.2f}")

        elif choice == "2":
            dev_data = load_dev_data("dev.json")
            print(f"\nAm găsit {len(dev_data)} texte în dev.json. Începem predictiile...\n")

            for i, text in enumerate(dev_data[:10], 1):  # Limităm la 10 exemple pentru vizualizare rapidă
                simplified_output = pipeline.simplify(text, apply_postprocess=True)
                bleu_score = calculate_bleu(text, simplified_output)

                print(f"Text {i}:")
                print("Original text:")
                print(format_text(text))
                print("\nSimplified text:")
                print(format_text(simplified_output))
                print(f"\nBLEU Score: {bleu_score:.2f}")
                print("-" * 80)

        elif choice == "3":
            print("\nLa revedere!")
            break

        else:
            print("\nOpțiune invalidă. Te rog să alegi 1, 2 sau 3.")

if __name__ == "__main__":
    main()
