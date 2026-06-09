import gradio as gr
from src.predict import load_classifier, predict_sentiment

MODEL_ID = "your-hf-username/sentiment-demo-model"

classifier = load_classifier(MODEL_ID)

def predict(text):
    return predict_sentiment(classifier, text)

demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=4, placeholder="Enter review text..."),
    outputs=gr.JSON(),
    title="Sentiment Classifier",
    description="Model served from Hugging Face Hub."
)

demo.launch()
