import re
import fasttext
import fasttext.util

class TextPreprocessor:
    def __init__(self, fasttext_model_path: str = None):
        """
        Dacă vrei să folosești fastText, indică calea către fișierul .bin (ex: 'cc.en.300.bin').
        """
        self.ft_model = None
        if fasttext_model_path is not None:
            print("Loading fastText model. This might take a while...")
            self.ft_model = fasttext.load_model(fasttext_model_path)
            print("fastText model loaded.")
        else:
            print("No fastText model path provided. Post-processing synonyms may be limited.")

    def clean_text(self, text: str) -> str:
        """
        Curăță textul de caractere nedorite, spații multiple etc.
        """
        text = text.strip()
        # Eliminăm caractere non-ASCII
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        # Ex.: lăsăm litere, cifre, semne de punctuație standard
        # Eliminăm excesul de spații
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def get_word_vector(self, word: str):
        if self.ft_model is None:
            return None
        return self.ft_model.get_word_vector(word)

    def tokenize(self, text: str):
        """
        Tokenizare simplă (opțional).
        """
        return text.split()

if __name__ == "__main__":
    # Test
    preprocessor = TextPreprocessor(fasttext_model_path=None)
    text = "Jeddah is the principal gateway to Mecca, Islam's holiest city."
    cleaned = preprocessor.clean_text(text)
    print("Cleaned text:", cleaned)
