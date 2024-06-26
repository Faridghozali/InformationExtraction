import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK stopwords if not already downloaded
try:
    nltk.data.find('corpora/stopwords.zip')
except:
    nltk.download('stopwords')

# Load stopwords for Indonesian
stop_words = stopwords.words('indonesian')

# Function to preprocess Indonesian text
def preprocess_text_indonesia(text):
    text = text.lower()  # Lowercasing
    tokens = word_tokenize(text)  # Tokenization
    tokens = [word for word in tokens if word.isalnum()]  # Remove punctuation
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    preprocessed_text = ' '.join(tokens)  # Join tokens back into text
    return preprocessed_text

# Function to load dataset
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Function to extract n-grams
def extract_ngrams(texts, ngram_range=(1, 2)):
    vectorizer = CountVectorizer(ngram_range=ngram_range, stop_words=None)
    X = vectorizer.fit_transform(texts)
    ngrams = vectorizer.get_feature_names_out()
    counts = X.toarray().sum(axis=0)
    df_ngrams = pd.DataFrame({'ngram': ngrams, 'count': counts})
    df_ngrams = df_ngrams.sort_values(by='count', ascending=False)
    return df_ngrams

# Main title of the application
st.title('Ekstraksi Pola Ujaran Kebencian')

# Sidebar with additional tabs
with st.sidebar:
    st.subheader('Menu')
    selected_tab = st.radio('Pilih Tab:', ('Ekstraksi N-gram', 'Data dan Penjelasan'))

# Main content based on selected tab
if selected_tab == 'Ekstraksi N-gram':
    user_input = st.text_area("Masukkan teks yang ingin dianalisis:", "")

    if user_input:
        preprocessed_text = preprocess_text_indonesia(user_input)
        texts = preprocessed_text.split('\n')
        df_ngrams = extract_ngrams(texts)

        st.subheader('Frekuensi N-gram')
        st.dataframe(df_ngrams)

        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(df_ngrams.set_index('ngram').to_dict()['count'])

        st.subheader('WordCloud N-gram')
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)

elif selected_tab == 'Data dan Penjelasan':
    st.subheader('Data dan Penjelasan')
    st.markdown("""
    Di tab ini, Anda dapat menampilkan semua data yang relevan dan penjelasan terkait analisis atau hasil dari ekstraksi pola ujaran kebencian.
    
    ### Tabel Dataset: DATASET CYBERBULLYING INSTAGRAM - FINAL
    """)

    df_dataset = load_data('DATASET CYBERBULLYING INSTAGRAM - FINAL.csv')
    
    st.dataframe(df_dataset)

    st.markdown("""
    ### Penjelasan Dataset
    Anda dapat menambahkan penjelasan tambahan tentang dataset ini di sini.
    """)
