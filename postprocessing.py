import numpy as np
import nltk
from nltk.corpus import wordnet

nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

class FastTextPostProcessor:
    def __init__(self, preprocessor, frequency_threshold=20000):
        """
        preprocessor: instanță de TextPreprocessor cu model fastText încărcat.
        frequency_threshold: considerăm un cuvânt 'complex' dacă frecvența e sub acest prag
        (aici e un exemplu minimal, folosind un fake dictionary).
        """
        self.preprocessor = preprocessor
        self.frequency_threshold = frequency_threshold
        # Dicționar de frecvențe (foarte simplificat). Trebuie extins cu date reale.
        self.fake_freq_dict = {
            'janjaweed': 1000,
            'holiest': 1000,
            'principal': 15000,
            'gateway': 12000,
            'armed': 25000,
            'existence': 15000,
            'methane': 5000,
            'conflicts': 40000,
            # ... completezi cu ce ai nevoie ...
        }

    def is_complex(self, word: str) -> bool:
        freq = self.fake_freq_dict.get(word.lower(), 5000)  # default
        return freq < self.frequency_threshold

    def get_synonyms(self, word: str) -> list:
        """
        Găsește sinonime posibile folosind WordNet.
        """
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                # Eliminăm underscore
                lemma_name = lemma.name().lower().replace('_', ' ')
                synonyms.add(lemma_name)
        # Eliminăm cuvântul original
        synonyms.discard(word.lower())
        return list(synonyms)

    def cosine_similarity(self, vec1, vec2):
        dot = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0
        return dot / (norm1 * norm2)

    def choose_best_synonym(self, original_word, synonyms):
        """
        Alege sinonimul cel mai apropiat semantic de original_word,
        pe baza embedding-urilor fastText.
        """
        if not self.preprocessor.ft_model:
            # Dacă nu avem model, returnăm direct primul sinonim
            return synonyms[0] if synonyms else original_word

        original_vec = self.preprocessor.get_word_vector(original_word)
        if original_vec is None:
            return original_word

        best_syn = original_word
        best_score = -1.0
        for syn in synonyms:
            syn_vec = self.preprocessor.get_word_vector(syn)
            if syn_vec is not None:
                score = self.cosine_similarity(original_vec, syn_vec)
                if score > best_score:
                    best_score = score
                    best_syn = syn
        return best_syn

    def process_text(self, text: str) -> str:
        """
        1. Tokenizăm textul.
        2. Pentru fiecare cuvânt 'complex', căutăm sinonime și alegem cel mai apropiat semantic.
        3. Reconstruim textul.
        """
        tokens = text.split()
        new_tokens = []
        for t in tokens:
            # verificăm dacă e un cuvânt 'alfabetic'
            if t.isalpha() and self.is_complex(t):
                syns = self.get_synonyms(t)
                if syns:
                    best_syn = self.choose_best_synonym(t, syns)
                    new_tokens.append(best_syn)
                else:
                    new_tokens.append(t)
            else:
                new_tokens.append(t)
        return " ".join(new_tokens)