import os
import yaml
import numpy as np
import evaluate
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
    config = load_config()

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable is missing")

    login(token=hf_token)

    dataset = load_dataset(config["dataset_name"])

    train_data = dataset["train"].shuffle(seed=42).select(
        range(config["train_samples"])
    )
    eval_data = dataset["test"].shuffle(seed=42).select(
        range(config["eval_samples"])
    )

    tokenizer = AutoTokenizer.from_pretrained(config["base_model"])

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=config["max_length"]
        )

    train_data = train_data.map(tokenize, batched=True)
    eval_data = eval_data.map(tokenize, batched=True)

    train_data = train_data.rename_column("label", "labels")
    eval_data = eval_data.rename_column("label", "labels")

    train_data.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    eval_data.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    model = AutoModelForSequenceClassification.from_pretrained(
        config["base_model"],
        num_labels=2
    )

    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return accuracy.compute(predictions=predictions, references=labels)

    training_args = TrainingArguments(
        output_dir="./outputs",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        num_train_epochs=config["epochs"],
        weight_decay=0.01,
        push_to_hub=True,
        hub_model_id=config["hf_repo"],
        hub_token=hf_token,
        logging_dir="./logs"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        compute_metrics=compute_metrics
    )

    trainer.train()
    results = trainer.evaluate()

    print("Evaluation results:", results)

    trainer.push_to_hub(commit_message="Upload trained model")
    tokenizer.push_to_hub(config["hf_repo"])

if __name__ == "__main__":
    main()