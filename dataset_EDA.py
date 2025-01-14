import json
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import seaborn as sns
import string

def load_dataset(file_path):
    """
    Load dataset from a JSON file.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def text_length_statistics(data):
    """
    Compute and display text length statistics for complex and simplified texts.
    """
    complex_lengths = [len(entry["complex_text"].split()) for entry in data]
    simple_lengths = [len(entry["simple_text"].split()) for entry in data]

    print("\nText Length Statistics:")
    print(f"Complex Texts: Min={np.min(complex_lengths)}, Max={np.max(complex_lengths)}, "
          f"Mean={np.mean(complex_lengths):.2f}, Median={np.median(complex_lengths)}")
    print(f"Simplified Texts: Min={np.min(simple_lengths)}, Max={np.max(simple_lengths)}, "
          f"Mean={np.mean(simple_lengths):.2f}, Median={np.median(simple_lengths)}")

    return complex_lengths, simple_lengths

def plot_text_length_distribution(complex_lengths, simple_lengths):
    """
    Plot histogram of text lengths for complex and simplified texts.
    """
    plt.figure(figsize=(12, 6))
    sns.histplot(complex_lengths, label="Complex Texts", color="blue", kde=True, stat="density", bins=30)
    sns.histplot(simple_lengths, label="Simplified Texts", color="green", kde=True, stat="density", bins=30)
    plt.title("Text Length Distribution")
    plt.xlabel("Number of Words")
    plt.ylabel("Density")
    plt.legend()
    plt.show()

def vocabulary_analysis(data):
    """
    Analyze the vocabulary of the dataset, excluding punctuation and normalizing case.
    """
    punctuation = set(string.punctuation)  # Set de caractere de punctuație
    complex_vocab = Counter(
        word.lower() for entry in data
        for word in entry["complex_text"].split()
        if word not in punctuation
    )
    simple_vocab = Counter(
        word.lower() for entry in data
        for word in entry["simple_text"].split()
        if word not in punctuation
    )

    print("\nVocabulary Analysis (Excluding Punctuation and Case-Insensitive):")
    print(f"Unique Words in Complex Texts: {len(complex_vocab)}")
    print(f"Unique Words in Simplified Texts: {len(simple_vocab)}")

    # Most common words
    print("\nMost Common Words in Complex Texts:")
    print(complex_vocab.most_common(10))
    print("\nMost Common Words in Simplified Texts:")
    print(simple_vocab.most_common(10))

    return complex_vocab, simple_vocab

def plot_word_frequency(vocab, title):
    """
    Plot the frequency of the most common words.
    """
    most_common = vocab.most_common(10)
    words, counts = zip(*most_common)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=counts, y=words, palette="viridis")
    plt.title(title)
    plt.xlabel("Frequency")
    plt.ylabel("Words")
    plt.show()

def main():
    dataset_path = "train.json"  # Update if needed
    data = load_dataset(dataset_path)

    # Text Length Analysis
    complex_lengths, simple_lengths = text_length_statistics(data)
    plot_text_length_distribution(complex_lengths, simple_lengths)

    # Vocabulary Analysis
    complex_vocab, simple_vocab = vocabulary_analysis(data)
    plot_word_frequency(complex_vocab, "Top 10 Words in Complex Texts")
    plot_word_frequency(simple_vocab, "Top 10 Words in Simplified Texts")

if __name__ == "__main__":
    main()
