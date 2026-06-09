import os
import yaml
import numpy as np
from datasets import load_dataset
from huggingface_hub import login
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)


def load_config(path="configs/train_config.yaml"):
    with open(path, "r") as file:
        return yaml.safe_load(file)


def main():
    # Load training config
    config = load_config()

    # Read Hugging Face token from environment variable
    hf_token = os.environ.get("HF_TOKEN")

    if not hf_token:
        raise ValueError(
            "HF_TOKEN environment variable is missing. "
            "Please set it in Kaggle Secrets and load it before running training."
        )

    # Login to Hugging Face
    login(token=hf_token)

    # Load dataset
    dataset = load_dataset(config["dataset_name"])

    # Use small subset to save Kaggle GPU time
    train_data = dataset["train"].shuffle(seed=42).select(
        range(config["train_samples"])
    )

    eval_data = dataset["test"].shuffle(seed=42).select(
        range(config["eval_samples"])
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["base_model"])

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=config["max_length"]
        )

    # Tokenize dataset
    train_data = train_data.map(tokenize, batched=True)
    eval_data = eval_data.map(tokenize, batched=True)

    # Rename label column to labels because Trainer expects "labels"
    train_data = train_data.rename_column("label", "labels")
    eval_data = eval_data.rename_column("label", "labels")

    # Set PyTorch format
    train_data.set_format(
        "torch",
        columns=["input_ids", "attention_mask", "labels"]
    )

    eval_data.set_format(
        "torch",
        columns=["input_ids", "attention_mask", "labels"]
    )

    # Load model for classification
    model = AutoModelForSequenceClassification.from_pretrained(
        config["base_model"],
        num_labels=2
    )

    # Simple accuracy calculation without using evaluate package
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = (predictions == labels).mean()
        return {"accuracy": float(accuracy)}

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./outputs",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        num_train_epochs=config["epochs"],
        weight_decay=0.01,
        logging_dir="./logs",
        push_to_hub=True,
        hub_model_id=config["hf_repo"],
        hub_token=hf_token,
        report_to="none"
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        compute_metrics=compute_metrics
    )

    # Train model
    trainer.train()

    # Evaluate model
    results = trainer.evaluate()
    print("Evaluation results:")
    print(results)

    # Push model and tokenizer to Hugging Face Hub
    trainer.push_to_hub(commit_message="Upload trained sentiment model")
    tokenizer.push_to_hub(config["hf_repo"])

    print("Model uploaded successfully to Hugging Face:")
    print(config["hf_repo"])


if __name__ == "__main__":
    main()
