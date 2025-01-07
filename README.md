# Text Simplification System

This project is a system for **text simplification in English**, leveraging a fine-tuned **T5 model** combined with **fastText** for post-processing. The aim is to reduce the linguistic complexity of texts while preserving their meaning and original information. The project is designed for **educational purposes**, **research**, or creating **accessible solutions** for users who struggle with understanding complex language.

## Technologies Used
- **Python 3.11**: The primary programming language.
- **Hugging Face Transformers**: For managing and fine-tuning the T5 model.
- **PyTorch**: The core framework used for training the model.
- **fastText**: Used to select appropriate synonyms during post-processing.
- **Datasets**: For manipulating and preprocessing datasets.
- **Evaluate**: To compute BLEU and SARI metrics.
- **JSON**: For storing and loading training and testing datasets.

## Project Structure

### `dataset.py`
- Handles loading datasets from JSON files.
- Includes functions for loading and preprocessing datasets for training and evaluation.

### `datasetmaker.py` *(Optional)*
- A utility for creating new datasets from raw files.

### `preprocessing.py`
- Contains logic for text preprocessing, including cleaning, tokenization, and preparing for fastText.

### `train.py`
- A script to fine-tune the **T5 model** on the text simplification dataset.
- Allows adjustments to training parameters, such as the number of epochs, batch size, and learning rate.

### `postprocessing.py`
- Implements logic to replace complex words with appropriate synonyms using **fastText**.

### `pipeline.py`
- The main class that integrates the **T5 model** and post-processing logic into a simplified pipeline.

### `predict.py`
- A script for inference.
- Allows users to:
  - Input texts manually for simplification.
  - Run predictions on a subset of texts from `dev.json`.

### `evaluate_local.py`
- A script to evaluate the model using **BLEU** and **SARI** metrics.
- Outputs the model's performance on the test dataset.
