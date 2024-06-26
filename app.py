import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.util import ngrams
from collections import Counter
import re

# Function to clean text
def clean_text(text):
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove punctuation and numbers
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    # Lowercase
    text = text.lower()
    return text

# Function to tokenize and remove stopwords
def preprocess_text(text):
    stop_words = set(stopwords.words('english')) # Change to appropriate language
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in stop_words]
    return tokens

# Function to extract n-grams
def extract_ngrams(tokens, n):
    n_grams = ngrams(tokens, n)
    return [' '.join(grams) for grams in n_grams]

# Function to plot word cloud (not implemented in this example)
def plot_word_cloud(text_data):
    # Generate word cloud or other visualizations
    pass

# Main function for Streamlit app
def main():
    st.title('Hate Speech Pattern Extraction and Analysis')

    # Sidebar for user input or file upload
    st.sidebar.title('Upload or Input Text Data')
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head())

        # Display available column names for debugging
        st.sidebar.subheader('Available Columns:')
        st.sidebar.write(df.columns.tolist())

        # Allow user to select the column containing text data
        text_column_name = st.sidebar.selectbox("Select Column Containing Text Data", df.columns.tolist())

        # Proceed only if a column is selected
        if text_column_name:
            # Clean and preprocess text
            df['clean_text'] = df[text_column_name].apply(clean_text)
            df['tokens'] = df['clean_text'].apply(preprocess_text)

            # Extract n-grams
            n = st.sidebar.number_input("Select n for n-grams", min_value=1, max_value=3, value=2)
            df['ngrams'] = df['tokens'].apply(lambda x: extract_ngrams(x, n))

            # Analyze frequencies and patterns
            st.subheader('Top n-grams')
            all_ngrams = [ngram for ngrams_list in df['ngrams'] for ngram in ngrams_list]
            ngram_counter = Counter(all_ngrams)
            top_ngrams = ngram_counter.most_common(10)
            st.write(top_ngrams)

            # Plot word cloud or other visualizations
            st.subheader('Word Cloud')
            plot_word_cloud(df['clean_text'])

        else:
            st.sidebar.warning('Please select a column containing text data.')

    else:
        st.sidebar.info('Upload a CSV file to start')

if __name__ == '__main__':
    main()
