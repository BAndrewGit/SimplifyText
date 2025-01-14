import os
import re
import json
import unicodedata
from glob import glob
from sklearn.model_selection import train_test_split

def normalize_text(text):
    """
    Normalizează textul pentru a elimina caractere speciale neintenționate și codificările corupte.
    """
    text = unicodedata.normalize("NFC", text)

    replacements = {
        "â": "a",
        "Ã©": "é",
        "Ã": "à",
        "ë": "e",
        "ì": "i",
        "â": "—",
        "â¦": "...",
        "â": "\"",
        "â": "\"",
        "â": "'",
        "â¢": "*",
        "â¬": "€",
        "\\/": "/",
        "-LRB-": "(",  # Înlocuim -LRB- cu (
        "-RRB-": ")",  # Înlocuim -RRB- cu )
    }

    for wrong_char, correct_char in replacements.items():
        text = text.replace(wrong_char, correct_char)

    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text

def clean_text(text):
    """
    Curăță textul eliminând prefixele și normalizează spațiile.
    """
    cleaned_text = re.sub(r"^[^\t]+\t\d+\s+", "", text)
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()
    return cleaned_text

def is_valid_pair(complex_text, simple_text):
    """
    Verifică dacă perechea complex_text și simple_text este validă.
    """
    if (
        complex_text == simple_text or
        not simple_text.strip() or
        len(complex_text.split()) < 4 or
        len(simple_text.split()) < 4
    ):
        return False
    return True

def process_asset_dataset(folder_path):
    """
    Procesează dataset-ul ASSET și returnează perechi (complex_text, simple_text).
    """
    data = []

    orig_files = glob(os.path.join(folder_path, "asset.*.orig"))
    simp_files = sorted(glob(os.path.join(folder_path, "asset.*.simp.*")))

    for orig_file in orig_files:
        with open(orig_file, "r", encoding="utf-8") as orig_f:
            complex_lines = orig_f.readlines()

        for simp_file in simp_files:
            with open(simp_file, "r", encoding="utf-8") as simp_f:
                simple_lines = simp_f.readlines()

            if len(complex_lines) != len(simple_lines):
                print(f"Fișierele {orig_file} și {simp_file} nu au același număr de linii! Ignorat.")
                continue

            for complex_text, simple_text in zip(complex_lines, simple_lines):
                complex_text = normalize_text(clean_text(complex_text))
                simple_text = normalize_text(clean_text(simple_text))

                if is_valid_pair(complex_text, simple_text):
                    data.append({
                        "complex_text": complex_text,
                        "simple_text": simple_text
                    })

    print(f"ASSET dataset procesat: {len(data)} perechi.")
    return data

def process_wikisimple_dataset(normal_file, simple_file):
    """
    Procesează dataset-ul WikiSimple și returnează perechi (complex_text, simple_text).
    """
    data = []

    with open(normal_file, "r", encoding="utf-8") as normal_f, open(simple_file, "r", encoding="utf-8") as simple_f:
        normal_lines = normal_f.readlines()
        simple_lines = simple_f.readlines()

    if len(normal_lines) != len(simple_lines):
        print(f"Fișierele {normal_file} și {simple_file} nu au același număr de linii! Ignorat.")
        return data

    for normal_line, simple_line in zip(normal_lines, simple_lines):
        normal_text = normalize_text(clean_text(normal_line))
        simple_text = normalize_text(clean_text(simple_line))

        if is_valid_pair(normal_text, simple_text):
            data.append({
                "complex_text": normal_text,
                "simple_text": simple_text
            })

    print(f"WikiSimple dataset procesat: {len(data)} perechi.")
    return data

def split_and_save_data(data, train_output, dev_output, test_size=0.1):
    """
    Împarte datele în train și dev și salvează în fișiere JSON.
    """
    train_data, dev_data = train_test_split(data, test_size=test_size, random_state=42)

    with open(train_output, "w", encoding="utf-8") as train_f:
        json.dump(train_data, train_f, indent=4, ensure_ascii=False)
    print(f"Date de antrenament salvate în {train_output} ({len(train_data)} mostre).")

    with open(dev_output, "w", encoding="utf-8") as dev_f:
        json.dump(dev_data, dev_f, indent=4, ensure_ascii=False)
    print(f"Date de validare salvate în {dev_output} ({len(dev_data)} mostre).")

def process_all_datasets():
    """
    Funcție principală care procesează toate dataset-urile și generează train.json + dev.json.
    """
    asset_folder = "./raw/asset"
    wikisimple_doc_normal = "./raw/wikisimple/document-aligned/normal.txt"
    wikisimple_doc_simple = "./raw/wikisimple/document-aligned/simple.txt"
    wikisimple_sent_normal = "./raw/wikisimple/sentence-aligned/normal.aligned"
    wikisimple_sent_simple = "./raw/wikisimple/sentence-aligned/simple.aligned"

    train_output = "train.json"
    dev_output = "dev.json"

    all_data = []

    if os.path.exists(asset_folder):
        asset_data = process_asset_dataset(asset_folder)
        all_data.extend(asset_data)
    else:
        print(f"Folderul ASSET {asset_folder} nu există!")

    if os.path.exists(wikisimple_doc_normal) and os.path.exists(wikisimple_doc_simple):
        wikisimple_doc_data = process_wikisimple_dataset(wikisimple_doc_normal, wikisimple_doc_simple)
        all_data.extend(wikisimple_doc_data)
    else:
        print(f"Fișierele {wikisimple_doc_normal} sau {wikisimple_doc_simple} nu există!")

    if os.path.exists(wikisimple_sent_normal) and os.path.exists(wikisimple_sent_simple):
        wikisimple_sent_data = process_wikisimple_dataset(wikisimple_sent_normal, wikisimple_sent_simple)
        all_data.extend(wikisimple_sent_data)
    else:
        print(f"Fișierele {wikisimple_sent_normal} sau {wikisimple_sent_simple} nu există!")

    if all_data:
        split_and_save_data(all_data, train_output, dev_output)
    else:
        print("Nu există date pentru a genera train/dev!")

if __name__ == "__main__":
    process_all_datasets()
