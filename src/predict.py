from transformers import pipeline

def load_classifier(model_id):
    return pipeline(
        "text-classification",
        model=model_id,
        tokenizer=model_id
    )

def predict_sentiment(classifier, text):
    if not text or not text.strip():
        return {"error": "Input text is empty"}

    result = classifier(text)[0]

    return {
        "label": result["label"],
        "confidence": round(result["score"], 4)
    }