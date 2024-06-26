import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
    
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

# Function to generate word cloud
def generate_wordcloud(df_ngrams):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(df_ngrams.set_index('ngram').to_dict()['count'])
    return wordcloud

# Title of the application
st.title('Ekstraksi Pola Ujaran Kebencian')

# Sidebar with additional tabs
with st.sidebar:
    st.subheader('Menu')
    selected_tab = st.radio('Pilih Tab:', ('Data dan Penjelasan', 'Ekstraksi N-gram'))

# Main content based on selected tab
if selected_tab == 'Data dan Penjelasan':
    st.subheader('Data dan Penjelasan')
    st.markdown("""
    Di tab ini, Anda dapat menampilkan semua data yang relevan dan penjelasan terkait analisis atau hasil dari ekstraksi pola ujaran kebencian.
    
    ### Tabel Dataset: DATASET CYBERBULLYING INSTAGRAM - FINAL
    """)

    # Load dataset
    df_dataset = load_data('DATASET CYBERBULLYING INSTAGRAM - FINAL.csv')

    # Display dataset table
    st.dataframe(df_dataset)

    st.markdown("""
    ### Data yang sudah di preprocessing
    """)

    # Load preprocessed data (if available)
    df = load_data('DataPba.csv')
    st.dataframe(df)

elif selected_tab == 'Ekstraksi N-gram':
    st.subheader('Ekstraksi N-gram')
    # User input text
    user_input = st.text_area("Masukkan teks yang ingin dianalisis:", "")

    if user_input:
        # Split input into sentences
        texts = user_input.split('\n')

        # Extract n-grams
        df_ngrams = extract_ngrams(texts)

        # Display n-gram frequencies as a table
        st.subheader('Frekuensi N-gram')
        st.dataframe(df_ngrams)

        # Generate and display word cloud
        st.subheader('WordCloud N-gram')
        wordcloud = generate_wordcloud(df_ngrams)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)

        # Generate and display histogram for top 20 n-grams
        st.subheader('Histogram Top 20 N-gram')
        plt.figure(figsize=(12, 6))
        plt.bar(df_ngrams['ngram'].head(20), df_ngrams['count'].head(20))
        plt.xticks(rotation=45, ha='right')
        plt.xlabel('N-gram')
        plt.ylabel('Count')
        st.pyplot(plt)
