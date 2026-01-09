import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import numpy as np

class EmotionClassifier:
    def __init__(self, model_name='bhadresh-savani/distilbert-base-uncased-emotion'):
        print(f"Loading emotion model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
        
        # Ensure model is in eval mode
        self.model.eval()

    def predict(self, text):
        """
        Returns a dictionary of emotion probabilities and the raw probability vector.
        """
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Apply softmax to get probabilities
        probs = F.softmax(outputs.logits, dim=1).numpy()[0]
        
        # Map to labels
        result = {label: float(prob) for label, prob in zip(self.labels, probs)}
        
        # Sort by probability
        sorted_result = dict(sorted(result.items(), key=lambda item: item[1], reverse=True))
        
        return sorted_result, probs

    def get_labels(self):
        return self.labels

if __name__ == "__main__":
    # Test the classifier
    clf = EmotionClassifier()
    text = "I feel a bit overwhelmed but also excited about the new project."
    res, vec = clf.predict(text)
    print(f"Input: {text}")
    print(f"Result: {res}")
    print(f"Vector: {vec}")
