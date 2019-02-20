#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 12:11:52 2019

Creating sentiment analysis for movie reviews

@author: Pran Kumar Sarkar
"""
# Importing datasets
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.datasets import load_files
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import pickle

"""
# Loading datasets (for first time only)
dataset = load_files('reviews')
features, labels = dataset.data, dataset.target

# Persisting the datasets to save time on furthur load
with open('features.pickle', 'wb') as f:
    pickle.dump(features, f)

with open('labels.pickle', 'wb') as f:
    pickle.dump(labels, f)
"""

# Loading dataset (Unpickling datasets)
with open('features.pickle', 'rb') as f:
    features = pickle.load(f)
    
with open('labels.pickle', 'rb') as f:
    labels = pickle.load(f)

# Preprocessing data
corpus = []
stopwords = stopwords.words('english')

for i in range(len(features)):
    
    # Removing all punctuation marks and non characters
    review = re.sub(r'\W', ' ', str(features[i]))
    
    # Converting into lowercase
    review = review.lower()
    
    # Removing b from starting of string
    review = re.sub(r'^b\s+', '', review)
    
    # Removing all single characters
    review = re.sub(r'\s+[a-z]\s+', ' ', review)
    
    # Removing all words which is of length one
    review = re.sub(r'[^a-z]\s+', ' ', review)
    
    # Removing all extra spaces
    review = re.sub(r'\s+', ' ', review)
    
    # Adding cleaned reviews in corpus
    corpus.append(review)

# Creating Tfidf model
vectorizer = TfidfVectorizer(max_features=2000,
                             min_df=3,
                             max_df=.6,
                             stop_words=stopwords)
tf_idf_model = vectorizer.fit_transform(corpus).toarray()

# Splitting dataset into train and test sets
feature_train, feature_test, labels_train, labels_test = train_test_split(tf_idf_model,
                                                                          labels,
                                                                          test_size=.2,
                                                                          random_state=0)

# Training Logistic Regression model
classifier = LogisticRegression()
classifier.fit(feature_train, labels_train)

# Searching for best hyperparameters using GridSearchCV
parameters = dict(C=[.5,1,2,3,4,5,6,7,8,9,10,15])

# Training GridSearch Model
grid_model = GridSearchCV(classifier,
                          param_grid=parameters,
                          n_jobs=-1,
                          cv=10,
                          scoring='accuracy')
grid_model.fit(feature_train, labels_train)

# Displaying best hyperparameters
print('Accuracy :', grid_model.best_score_)
print('Best Parameters :', grid_model.best_params_)
print('Best Estimator :\n', grid_model.best_estimator_)

# Training our logistic regression model with best hyperparameters
classifier = LogisticRegression(C=5, class_weight=None, dual=False, fit_intercept=True,
                                intercept_scaling=1, max_iter=100, multi_class='warn',
                                n_jobs=None, penalty='l2', random_state=None, solver='warn',
                                tol=0.0001, verbose=0, warm_start=False)
classifier.fit(feature_train, labels_train)

# Predicting values
labels_pred = classifier.predict(feature_test)

# Testing model performance
print('Confusion matrix: \n', confusion_matrix(labels_test, labels_pred))
print('Accuracy: ', accuracy_score(labels_test, labels_pred))

# Saving Tfidf model and vectorizer
with open('vectorizer.pickle', 'wb') as f:
    pickle.dump(vectorizer, f)
    
with open('tfidfmodel.pickle', 'wb') as f:
    pickle.dump(classifier, f)

# Reloading classifier and testing our model
text = ['This is a very bad model and this is awesome.']

with open('vectorizer.pickle', 'rb') as f:
   vectorizer = pickle.load(f)
    
with open('tfidfmodel.pickle', 'rb') as f:
    tf_idf_model = pickle.load(f)

# Testing model
print(tf_idf_model.predict(vectorizer.transform(text).toarray()))

















