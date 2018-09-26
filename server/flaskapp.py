
# import dependencies, model, 

contractions = { 
"ain't": "am not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he's": "he is",
"how'd": "how did",
"how'll": "how will",
"how's": "how is",
"i'd": "i would",
"i'll": "i will",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'll": "it will",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"must've": "must have",
"mustn't": "must not",
"needn't": "need not",
"oughtn't": "ought not",
"shan't": "shall not",
"sha'n't": "shall not",
"she'd": "she would",
"she'll": "she will",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"that'd": "that would",
"that's": "that is",
"there'd": "there had",
"there's": "there is",
"they'd": "they would",
"they'll": "they will",
"they're": "they are",
"they've": "they have",
"wasn't": "was not",
"we'd": "we would",
"we'll": "we will",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"where'd": "where did",
"where's": "where is",
"who'll": "who will",
"who's": "who is",
"won't": "will not",
"wouldn't": "would not",
"you'd": "you would",
"you'll": "you will",
"you're": "you are"
}

import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
import re
# from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
# from keras.models import Sequential
# from keras.layers import Dense,GlobalAveragePooling1D
from keras.layers.embeddings import Embedding
from keras.layers import Convolution1D, Flatten, Dropout,MaxPooling1D
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences


nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

ps = nltk.PorterStemmer()

# we need to pickle the tokenization. the dataset is too large for github. 
def tokenize(bigText):
    top_words = 15000
    tokenizer = Tokenizer(num_words=top_words)
    tokenizer.fit_on_texts(bigText)
    return tokenizer

bigText = pd.read_csv('./600kReviewsOnly.csv')

tokenizerTexter = tokenize(bigText)
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

    single_tokenized_text = tokenizerTexter.texts_to_sequences(text)
    pad = 'pre'
    max_tokens = 444
    single_text_pad = pad_sequences(single_tokenized_text, maxlen=max_tokens,
                            padding=pad, truncating=pad)

    return single_text_pad
    


padded_tokenized_text = prepro_small()

import flask
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/thegender')
padded_text = prepro_small(text_in)
gender_guess = engine.predict(padded_text)
return gender_guess