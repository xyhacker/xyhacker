from flask import Flask,render_template, request
import re
import sys
import os
sys.path.append(os.path.abspath('./model'))
from load import *
import numpy as np
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences


#init flask app

application = Flask(__name__)
global model,graph
model,graph = init()
def convertData(x):
	top_words = 15000
	tokenizer = Tokenizer(num_words = top_words)
	tokenizer.fit_on_texts(x)
	max_tokens = 228
	x_train_tokens  = tokenizer.texts_to_sequences(x)
	x_train_pad = pad_sequences(x_train_tokens, maxlen=max_tokens,padding='pre', truncating='pre')
	return x_train_pad


@application.route('/')
def index():
	#initModel()
	#render out pre-built HTML file right on the index page
	return render_template("index.html")	

@application.route('/predict/',methods=['GET','POST'])
def predict():
	if request.method == 'POST':
		comment = request.form['comment']
		data = comment
		with graph.as_default():
			pred = model.predict(convertData(data))
			
			out = (-pred).argsort()[:1]
			
		return render_template('results.html',prediction = out,comment = comment)

    
	


if __name__ == '__main__':
	application.run()    


