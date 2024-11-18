from flask import Flask, request, render_template, flash, redirect
import pickle
import re
import logging
import os
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

app = Flask(__name__)

# Use a secure secret key, recommended to set in environment variables for production
app.secret_key = os.getenv('SECRET_KEY', 'default_secret_key')

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load the model and vectorizer with error handling
try:
    with open('best_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    logging.info("Model and vectorizer loaded successfully.")
except Exception as e:
    model, vectorizer = None, None
    logging.error(f"Error loading model or vectorizer: {e}")

lemmatizer = WordNetLemmatizer()

# Preprocessing function
def lemmatizing(content):
    content = re.sub(r'http\S+|www\S+|https\S+', '', content, flags=re.MULTILINE)
    content = re.sub(r'@\w+', '', content)
    content = re.sub('[^a-zA-Z]', ' ', content)
    content = content.lower()
    content = content.split()
    content = [lemmatizer.lemmatize(word) for word in content if word not in stopwords.words('english')]
    return ' '.join(content)

@app.route('/')
def home():
    return render_template('index.html', prediction_text=None)

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not vectorizer:
        flash("Model or vectorizer not loaded. Please contact the administrator.", "error")
        return redirect('/')

    if request.method == 'POST':
        message = request.form['message']

        # Check for empty message
        if not message.strip():
            flash("Please enter a message to predict.", "warning")
            return redirect('/')

        # Removed the length check for 160 characters
        # The following lines are no longer needed:
        # if len(message) > 160:
        #     flash("The message is too long (more than 160 characters). Please shorten it and try again.", "warning")
        #     return redirect('/')

        # Preprocess and predict
        processed_message = lemmatizing(message)
        message_vector = vectorizer.transform([processed_message])
        prediction = model.predict(message_vector)[0]
        label = "Spam" if prediction == 1 else "Ham"
        
        # Set prediction text based on the result
        prediction_text = f"Predicted Label: {label}"
        
        return render_template('index.html', prediction_text=prediction_text)

# Removed feedback-related routes

if __name__ == "__main__":
    # Determine if debug mode should be enabled from the environment
    debug_mode = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    
    # Run the app on the specified host and port
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=debug_mode)