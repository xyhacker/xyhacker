
# import dependencies, model, 

import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
import re
import pickle
# from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
# from keras.models import Sequential
# from keras.layers import Dense,GlobalAveragePooling1D
from keras.layers.embeddings import Embedding
from keras.layers import Convolution1D, Flatten, Dropout,MaxPooling1D
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

from contractions import contractions



nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

ps = nltk.PorterStemmer()

# we need to pickle the tokenization. the dataset is too large for github. 
# def tokenize(bigText):
#     top_words = 15000
#     tokenizer = Tokenizer(num_words=top_words)
#     tokenizer.fit_on_texts(bigText)
#     return tokenizer

# bigText = pd.read_csv('./600kReviewsOnly.csv')
# tokenizerTexter = tokenize(bigText)

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


# preprocess single record
def prepro_small(singelReview,remove_stopwords=True):
    # Convert words to lower case
    text = singleReview.lower()
    # Replace contractions with their longer forms 
    if True:
        # We are not using "text.split()" here
        #since it is not fool proof, e.g. words followed by punctuations "Are you kidding?I think you aren't."
        text = re.findall(r"[\w']+", text)
        new_text = []
        for word in text:
            if word in contractions:
                new_text.append(contractions[word])
            else:
                new_text.append(word)
        text = " ".join(new_text)
    
    # Format words and remove unwanted characters
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)# remove links
    text = re.sub(r'\<a href', ' ', text)# remove html link tag
    text = re.sub(r'&amp;', '', text) 
    text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)
    text = re.sub(r'<br />', ' ', text)
    text = re.sub(r'\'', ' ', text)
    
    # Optionally, remove stop words
    if remove_stopwords:
        text = text.split()
        stops = set(stopwords.words("english"))
        text = [w for w in text if not w in stops]
        text = " ".join(text)

    single_tokenized_text = tokenizer.texts_to_sequences(text)
    pad = 'pre'
    max_tokens = 444
    single_text_pad = pad_sequences(single_tokenized_text, maxlen=max_tokens,
                            padding=pad, truncating=pad)

    return single_text_pad
    
# # Import model 
# from keras.models import model_from_json

# # Model reconstruction from JSON file
# with open('model.json', 'r') as f:
#     model = model_from_json(f.read())
    
# # Load weights into the new model
# model.load_weights('model.h5')

# Load entire model. The method above uses 2 steps. Which may also work.
from keras.models import load_model

model = load_model('model.h5')

padded_tokenized_text = prepro_small()

import flask
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/thegender')
padded_text = prepro_small(text_in)
gender_guess = engine.predict(padded_text)
return gender_guess