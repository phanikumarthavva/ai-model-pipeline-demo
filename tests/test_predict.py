from src.predict import predict_sentiment

class DummyClassifier:
    def __call__(self, text):
        return [{"label": "POSITIVE", "score": 0.99}]

def test_predict_sentiment():
    classifier = DummyClassifier()
    result = predict_sentiment(classifier, "This is great")

    assert result["label"] == "POSITIVE"
    assert result["confidence"] == 0.99

def test_empty_input():
    classifier = DummyClassifier()
    result = predict_sentiment(classifier, "")

    assert "error" in result