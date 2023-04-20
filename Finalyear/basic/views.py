import nltk as nltk
from django.http import HttpResponse
from django.shortcuts import render
import regex as re
# from django.http import HttpResponse
import pandas as pd
import numpy as np
from nltk import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load the CSV file into a Pandas DataFrame
data = pd.read_csv('my_randomly_dropped_dataset.csv')

# Extract the 'text' and 'sentiment' columns from the DataFrame
X = data['Text']
y = data['Score']
for row in data.itertuples():
    if data.at[row.Index, "Score"] > 3:
        data.at[row.Index, "Score"] = 1
    else:
        data.at[row.Index, "Score"] = 0

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert the text data into numerical features using CountVectorizer
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Create a Gaussian Naive Bayes model
gnb = GaussianNB()

# Train the model on the training data
gnb.fit(X_train.toarray(), y_train)

# Make predictions on the testing data
y_pred = gnb.predict(X_test.toarray())

accuracy = accuracy_score(y_test, y_pred)
print(y_pred)
print('Accuracy:', accuracy)

# # Create your views here.
def index(request):
    return render(request, 'index.html')

def FormInfo(request):
        text_value = request.GET['inputText']
        prediction = gnb.predict(vectorizer.transform([text_value]).toarray())
        #print(prediction)
        # Print the sentiment prediction
        #c = 1
        if prediction == 0:
            output = "Positive sentiment"
            #c =0
        else:
            output = "Negative sentiment"
            #c += 1
        #print(prediction)
        # Do something with the text value
        return render(request, 'index.html', {'result': output})

