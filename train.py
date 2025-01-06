import numpy as np
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
from datasets import Dataset, DatasetDict
from dataset import load_train_dev_datasets
import torch


def train_t5_model(
    train_path="train.json",
    dev_path="dev.json",
    model_name="t5-base",
    output_dir="t5_simplification_model",
    num_train_epochs=2,
    batch_size=4,
    gradient_accumulation_steps=4,
    max_seq_length=64,
    lr=5e-5
):
    # Check GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Load datasets
    train_data, dev_data = load_train_dev_datasets(train_path, dev_path)
    train_data = train_data[:10000]
    dev_data = dev_data[:500]

    train_dataset = Dataset.from_list(train_data)
    dev_dataset = Dataset.from_list(dev_data)

    dataset_dict = DatasetDict({
        "train": train_dataset,
        "validation": dev_dataset
    })

    # Load model and tokenizer
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

    # Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)

    # Preprocess data
    def preprocess_function(examples):
        inputs = ["simplify: " + txt for txt in examples["complex_text"]]
        single_ref = [refs[0] if isinstance(refs, list) else refs for refs in examples["simple_text"]]
        model_inputs = tokenizer(inputs, max_length=max_seq_length, padding=True, truncation=True)
        labels = tokenizer(text_target=single_ref, max_length=max_seq_length, padding=True, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_dataset = dataset_dict.map(preprocess_function, batched=True)

    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        evaluation_strategy="no",
        save_strategy="epoch",
        save_total_limit=1,
        logging_steps=50,
        push_to_hub=False,
        learning_rate=lr,
        weight_decay=0.01,
        max_grad_norm=1.0,
        label_smoothing_factor=0.1,
        predict_with_generate=True,
        generation_max_length=64,
        generation_num_beams=4,
        fp16=True
    )

    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        data_collator=data_collator
    )

    # Train the model
    trainer.train()

    # Save the model
    trainer.save_model(output_dir)
    print(f"Model trained and saved in {output_dir}")


if __name__ == "__main__":
    train_t5_model(
        train_path="train.json",
        dev_path="dev.json",
        model_name="t5-base",
        output_dir="t5_simplification_model",
        num_train_epochs=2,
        batch_size=4,
        gradient_accumulation_steps=4,
        max_seq_length=64,
        lr=5e-5
    )
