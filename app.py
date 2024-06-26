import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Function to load dataset
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Function to generate WordCloud from all texts in the dataset
def generate_wordcloud(df, text_column='text'):
    all_texts = ' '.join(df[text_column].astype(str).tolist())
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_texts)
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

    # Generate and display WordCloud for all texts in dataset
    st.subheader('WordCloud Seluruh Teks Dataset')
    wordcloud_all_texts = generate_wordcloud(df_dataset, text_column='text')
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud_all_texts, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

    st.markdown("""
    ### Data yang sudah di preprocessing
    """)

    # Load preprocessed data if available
    df_preprocessed = load_data('DataPba.csv')
    st.dataframe(df_preprocessed)

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

        # Generate and display WordCloud for input text
        st.subheader('WordCloud N-gram')
        wordcloud = generate_wordcloud_from_ngrams(df_ngrams)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)
