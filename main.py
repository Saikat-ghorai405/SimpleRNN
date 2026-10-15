import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

## word index
word_index=imdb.get_word_index()
reverse_word_index={value :key for key,value in word_index.items()}


model=load_model('simple_rnn_imdb.h5')


def decode_review(encoded_review):
    return ' '.join([
        reverse_word_index.get(i-3, '?')
        for i in encoded_review
    ])

# Function to process user input
import re

import string

def preprocess_text(text, max_features=10000, maxlen=500):
    # 1. Strip punctuation so words like "fantastic!" match the dictionary
    clean_text = text.translate(str.maketrans('', '', string.punctuation))
    
    # 2. Lowercase and split into words (your sir's simple method)
    words = clean_text.lower().split()
    
    # 3. Start token [1] followed by encoded words with bounds checking
    encoded = [1]
    for w in words:
        idx = word_index.get(w)
        if idx is None or idx + 3 >= max_features:
            encoded.append(2)  # unknown / out of vocab
        else:
            encoded.append(idx + 3)
            
    # 4. Pad the sequence to match training length
    return sequence.pad_sequences([encoded], maxlen=maxlen)



import streamlit as st
## streamlit app

st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')

##user input
user_input=st.text_area('Movie Review')

if st.button('Classify'):
    preprocessed_input=preprocess_text(user_input)
    prediction=model.predict(preprocessed_input)
    sentiment='Positive' if prediction[0][0]>0.5 else 'Negative'

    ## display the result
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score: {prediction[0][0]}')
else:
    st.write('Please enter a moview review.')