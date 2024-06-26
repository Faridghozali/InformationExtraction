import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# Download NLTK resources if not already downloaded
try:
    nltk.data.find('corpora/stopwords.zip')
except:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)

# Load stopwords for English
stop_words = set(stopwords.words('english'))

# Function to preprocess text
def preprocess_text(text):
    # Lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]
    
    # Join tokens back into string
    preprocessed_text = ' '.join(tokens)
    
    return preprocessed_text

# Function to load dataset
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Function to extract n-grams
def extract_ngrams(texts, ngram_range=(1, 2)):
    vectorizer = CountVectorizer(ngram_range=ngram_range, stop_words='english')
    X = vectorizer.fit_transform(texts)
    ngrams = vectorizer.get_feature_names_out()
    counts = X.toarray().sum(axis=0)
    df_ngrams = pd.DataFrame({'ngram': ngrams, 'count': counts})
    df_ngrams = df_ngrams.sort_values(by='count', ascending=False)
    return df_ngrams

# Title of the application
st.title('Hate Speech Pattern Extraction')

# Sidebar with additional tabs
with st.sidebar:
    st.subheader('Menu')
    selected_tab = st.radio('Select Tab:', ('Extract N-grams', 'Data and Explanation'))

# Main content based on selected tab
if selected_tab == 'Extract N-grams':
    # User input text area
    user_input = st.text_area("Enter text to analyze:", "")

    if user_input:
        # Preprocess user input
        preprocessed_text = preprocess_text(user_input)
        
        # Split preprocessed text into sentences
        texts = preprocessed_text.split('\n')

        # Extract n-grams
        df_ngrams = extract_ngrams(texts)

        # Display n-gram table and frequencies
        st.subheader('N-gram Frequencies')
        st.dataframe(df_ngrams)

        # Generate WordCloud visualization
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(df_ngrams.set_index('ngram').to_dict()['count'])

        st.subheader('N-gram WordCloud')
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)

elif selected_tab == 'Data and Explanation':
    st.subheader('Data and Explanation')
    st.markdown("""
    In this tab, you can display all relevant data and explanations related to the analysis or results of hate speech pattern extraction.
    
    ### Dataset Table: DATASET CYBERBULLYING INSTAGRAM - FINAL
    """)

    # Load dataset
    df_dataset = load_data('DATASET CYBERBULLYING INSTAGRAM - FINAL.csv')
    
    # Display dataset table
    st.dataframe(df_dataset)

    st.markdown("""
    ### Dataset Explanation
    You can add additional explanations about this dataset here.
    """)
