# app.py

import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import spacy
from spacy import displacy
import gensim
import pyLDAvis.gensim as gensimvis
from collections import Counter
from tqdm.auto import tqdm
import time

# Load NLTK resources
nltk.download('stopwords')
stop_words = set(stopwords.words('indonesian'))

# Load dataset
df = pd.read_csv('DATASET.csv')

# Function to clean text
def clean_text(text):
    text = re.sub('<.*?>', '', text)  # Remove HTML tags
    text = re.sub('[^a-zA-Z]', ' ', text).lower()  # Remove non-alphabetic characters
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\S+', '', text)  # Remove mentions
    text = re.sub(r'#\S+', '', text)  # Remove hashtags
    
    words = nltk.word_tokenize(text)  # Tokenize
    words = [w for w in words if w not in stop_words]  # Remove stopwords
    stemmer = PorterStemmer()
    words = [stemmer.stem(w) for w in words]  # Stemming
    
    text = ' '.join(words)
    return text

# Clean text data
tqdm.pandas()
df['cleaned_text'] = df['Komentar'].progress_apply(clean_text)

# Create Bag of Words model
cv = CountVectorizer()
X = cv.fit_transform(df['cleaned_text']).toarray()
y = df['Kategori']

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# Function to predict and evaluate
def predict_evaluate(text):
    cleaned_text = clean_text(text)
    vectorized_text = cv.transform([cleaned_text]).toarray()
    prediction = clf.predict(vectorized_text)
    return prediction[0]

# Streamlit app
def main():
    st.title('Analisis Sentimen Komentar Instagram')
    
    # Sidebar with options
    st.sidebar.header('Options')
    task = st.sidebar.selectbox('Choose a task', ['Classify Comment', 'Show Data Analysis'])
    
    if task == 'Classify Comment':
        st.header('Classify Comment')
        comment = st.text_area('Input comment:')
        
        if st.button('Classify'):
            prediction = predict_evaluate(comment)
            st.success(f'This comment is classified as: {prediction}')
    
    elif task == 'Show Data Analysis':
        st.header('Data Analysis')
        st.subheader('Class Distribution')
        plt.figure(figsize=(8, 6))
        sns.countplot(x='Kategori', data=df)
        plt.title('Class Distribution of Cyberbullying')
        st.pyplot()
        
        st.subheader('Word Cloud')
        text = ' '.join(df['cleaned_text'])
        wordcloud = WordCloud(max_words=1000, width=800, height=400).generate(text)
        plt.figure(figsize=(10, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot()

# Run the app
if __name__ == '__main__':
    main()
