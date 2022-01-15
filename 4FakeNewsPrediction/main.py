import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import nltk

nltk.download('stopwords')
print(stopwords.words('english'))

news_dataset = pd.read_csv('data/train.csv')

# Check the number of rows and columns in dataset
print(news_dataset.shape)

# Print first 5 rows of the dataframe
print(news_dataset.head())

# Count number of nulls in each columns
print(news_dataset.isnull().sum())

# Replace null with empty string
news_dataset = news_dataset.fillna('')

# Combine author and title to create a new content column
news_dataset['content'] = news_dataset['author'] + ' ' + news_dataset['title']

# Extract data and label in separate dataframes
X = news_dataset.drop(columns='label', axis=1)
Y = news_dataset['label']

# Stemming process
# This process reduces a word to its Root word
# Example:
# Words like actor, actress, acting will be reduce to word act.
# The root word in all the words is act.
porterStemmer = PorterStemmer()


def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [porterStemmer.stem(word) for word in stemmed_content if word not in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content


# Apply stemming to content column
news_dataset['content'] = news_dataset['content'].apply(stemming)

# Extract content and label data in lists
X = news_dataset['content'].values
Y = news_dataset['label'].values

print(X)

# Converting textual data to numerical values
vectorizer = TfidfVectorizer()
vectorizer.fit(X)
X = vectorizer.transform(X)

print(X)

# Split into training and test datasets
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=1)

# Create Model
# Training a Model
model = LogisticRegression()
model.fit(X_Train, Y_Train)

# Evaluation
#   accuracy score
predicted_label = model.predict(X_Train)
score = accuracy_score(Y_Train, predicted_label)
print('Accuracy score of Training data: ', score)

# Let us now test model on Test Data
predicted_label = model.predict(X_Test)
score = accuracy_score(Y_Test, predicted_label)
print('Accuracy score of Test data: ', score)

# Creating a predictive system.
# Get new input
data_input = X_Test[0]
prediction = model.predict(data_input)

# Compare predicted value with actual value
print(prediction)
print(Y_Test[0])
