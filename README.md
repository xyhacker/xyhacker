# Amazon Text Analysis using NLTK
In this project we apply natural language processing techniques and ML models to predict the gender of Amazon grocery and gourmet food reviewers. 

As of 14/9/2018, we can predict the gender of the writer with __72.24% accuracy__, using the Keras CNN. Prior gender prediction research in the field has accomplished accuracy rates of 60-70%. 


#### Project Brief
Amazon has contracted your team to do an exploratory data analysis on product reviews. In particular, they are interested in being able to classify people as male or female based on their reviews. They have given you a dataset of customer reviews of grocery and gourmet food items. Create a model that identifies a person as male/female based on their review (regardless of product). 


# Dataset Used
[Grocery and Gourmet Food Dataset](http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Grocery_and_Gourmet_Food.json.gz)

[Amazon Dataset Guidelines](http://jmcauley.ucsd.edu/data/amazon/links.html)


# Methodology
The original Grocery and Gourmet food dataset does not include clear gender labels, only the names or usernames of the writers. Therefore, we used the [Gender Guesser Library](https://github.com/lead-ratings/gender-guesser) to label the data to be used in our prediction models. Using this method, we were able to classify roughly 25% of the samples available. Additional manual text processing got us to near full dataset to be labeled. 

We utilized various models for prediction, with results ranging from 50% to upwards of 70% for the best models. 


# Contributors
* Brenner Haverlock
* Kelvin Li
* Lorin Fields
* Saranya Mandava
* Tina Kovacova


This Capstone Project was developed by [Lambda School](https://lambdaschool.com/) Machine Learning and Data Science students.
