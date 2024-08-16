import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from nltk.stem import PorterStemmer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot
import pickle

app = Flask(__name__)

# Load the trained LSTM model
loaded_model = tf.keras.models.load_model('sentiment_model.h5') 

# Load preprocessing configuration
with open('preprocessing_config.pkl', 'rb') as f:
    config = pickle.load(f)

vocab_size = config['vocab_size']
sent_length = config['sent_length']

stemmer = PorterStemmer()

def preprocess_text(text):
    # Apply stemming to the input text
    tokens = text.split()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    return " ".join(stemmed_tokens)

def text_to_padded_sequence(text, vocab_size, sent_length):
    # Encode and pad the input message
    encoded_text = one_hot(text, vocab_size)
    padded_sequence = pad_sequences([encoded_text], maxlen=sent_length, padding='pre')
    return padded_sequence

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    
    data = request.get_json(force=True)
    text = data['text']
    
    
    preprocessed_text = preprocess_text(text)
    
   
    padded_sequence = text_to_padded_sequence(preprocessed_text, vocab_size, sent_length)

  
    pred = loaded_model.predict(padded_sequence)

   
    sentiment = "Positive" if pred[0][0] > 0.5 else "Negative"

    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run(debug=True)
