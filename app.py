import pickle
import numpy as np
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Load the saved model and vectorizer
with open("logistic_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("tfidf_vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Define the prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input JSON
        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"error": "Invalid input. Provide 'text' field in JSON."}), 400
        
        # Extract and transform text data
        text = data["text"]
        text_vectorized = vectorizer.transform([text])
        
        # Predict sentiment
        prediction = model.predict(text_vectorized)
        sentiment = "Positive" if prediction[0] == "Positive" else "Negative"
        
        # Return prediction
        return jsonify({"text": text, "sentiment": sentiment})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
