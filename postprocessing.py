import numpy as np
import nltk
from nltk.corpus import wordnet
import re

nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

class FastTextPostProcessor:
    def __init__(self, preprocessor, frequency_threshold=20000):
        self.preprocessor = preprocessor
        self.frequency_threshold = frequency_threshold
        self.fake_freq_dict = {
            'janjaweed': 1000,
            'holiest': 1000,
            'principal': 15000,
            'gateway': 12000,
            'armed': 25000,
            'existence': 15000,
            'methane': 5000,
            'conflicts': 40000,
        }
        self.preserved_phrases = {
            'artificial intelligence',
            'renewable energy',
            'natural language processing',
            'image recognition',
            'strategic planning'
        }

    def is_complex(self, word: str) -> bool:
        if len(word) < 4:
            return False
        freq = self.fake_freq_dict.get(word.lower(), self.frequency_threshold // 2)
        return freq < self.frequency_threshold

    def get_synonyms(self, word: str) -> list:
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                lemma_name = lemma.name().lower().replace('_', ' ')
                if self.is_valid_synonym(lemma_name, word):
                    synonyms.add(lemma_name)
        synonyms.discard(word.lower())
        return list(synonyms)

    def is_valid_synonym(self, synonym: str, original: str) -> bool:
        if len(synonym) > len(original) * 1.2:
            return False
        if not synonym.replace(" ", "").isalpha():
            return False
        return synonym.lower() != original.lower()

    def cosine_similarity(self, vec1, vec2):
        dot = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0
        return dot / (norm1 * norm2)

    def choose_best_synonym(self, original_word, synonyms):
        if not self.preprocessor.ft_model:
            return min(synonyms, key=len) if synonyms else original_word

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
        text = re.sub(r'\([^)]*\)', '', text)  # Remove text within parentheses
        text = re.sub(r'``', '', text)  # Remove opening double backticks
        text = re.sub(r"''", '', text)  # Remove closing double backticks

        tokens = text.split()
        new_tokens = []
        skip_replacement = False

        for i, t in enumerate(tokens):
            # Preserve phrases
            if any(phrase in text.lower() for phrase in self.preserved_phrases):
                skip_replacement = True

            if t[0].isupper() and i > 0 and not skip_replacement:
                new_tokens.append(t)
            elif t.isalpha() and self.is_complex(t):
                syns = self.get_synonyms(t)
                if syns:
                    best_syn = self.choose_best_synonym(t, syns)
                    if i == 0:
                        best_syn = best_syn.capitalize()
                    new_tokens.append(best_syn)
                else:
                    new_tokens.append(t)
            else:
                new_tokens.append(t)

        if new_tokens:
            new_tokens[0] = new_tokens[0].capitalize()

        return " ".join(new_tokens)
